import os, ast, json, time, random, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import yaml

from model import TGModel, load_ckpt, save_clusters
from load_data import (load_data, get_positive_weight, preprocess_to_dataset_normalized,
                       set_seed, summarise_towers, calculate_bbox_stats, calculate_and_print_metrics, reverse_calculation)
import load_data as _load_data
from utils import print_model_info, build_bank_optimized

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Test SymCAN model")
parser.add_argument("-c", "--config", default="configs/train.yaml",
                    help="Path to the yaml config used during training")
parser.add_argument("--run", default=None,
                    help="Run name (overrides run.name in yaml)")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

_load_data.init(cfg)

# ── Config values ─────────────────────────────────────────────────────────────
RUN_NAME          = cfg["run"]["name"]
SEED              = cfg["seed"]
TRAIN_VAL_SCRIPT  = cfg["data"]["csv_path"]
UNIQUE_DEPTH_DIR  = cfg["data"]["depth_image_dir"]
MAX_ROWS          = cfg["data"]["max_rows"]

DISCRETE          = cfg["model"]["discrete"]
SYMBOLIC          = cfg["model"]["symbolic"]
OUT_DIM           = 4
BB_DIM            = 4
SYM_SIZE          = cfg["model"]["symbol_size"]
OBJ_SYM_SIZE      = cfg["model"]["obj_symbol_size"]
COLLAPSE          = 1

W_MSE_REG         = cfg["loss"]["w_mse_reg"]
W_BCE             = cfg["loss"]["w_bce"]
W_MSE_DIMS        = torch.tensor(cfg["loss"]["w_mse_dims"])
ALPHA_VALS        = [1.0, 1.0, 1.0, 1.0]
THRESH_COLLAPSE   = 0.5
SCALE             = 100

REL_GRAPHS        = cfg["training"]["batch_size"]
NORMALIZATION     = True
MINMAX_SCALING    = False
GLOBAL            = False

set_seed(SEED)
LOG_DIR = os.path.join("logs", RUN_NAME)

# ── Setup ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Run: {RUN_NAME}  →  {LOG_DIR}")

mse_loss = nn.MSELoss(reduction='none')
# ── Test pass — hard Gumbel ───────────────────────────────────────────────────
def run_test(loader, model, pos_weight):
    model.eval()
    # Force hard discrete symbols for test
    model.gs_layer.hard     = True
    model.gs_layer.deterministic = True
    if hasattr(model, 'gs_obj_layer'):
        model.gs_obj_layer.hard          = True
        model.gs_obj_layer.deterministic = True

    total_loss = total_bce = total_mse = total_n = 0.0
    rec_pred, rec_true = [], []
    rec_rel_sym = []
    rec_query_obj_sym, rec_new_obj_sym = [], []   # ← per-relation query / new-object symbols
    rec_path, rec_obj, rec_q, rec_len, rec_bbox = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            given_nodes_list, new_object_node_list = [], []
            for i in range(batch.num_graphs):
                node_ids = (batch.batch == i).nonzero(as_tuple=True)[0]
                nodes_except_last = node_ids[:-1]
                given_nodes_list.append(nodes_except_last)
                new_object_node_list.append(torch.stack([node_ids[-1]] * len(nodes_except_last)))
            if not given_nodes_list: continue
            q_idx_batch       = torch.cat(given_nodes_list)
            new_obj_idx_batch = torch.cat(new_object_node_list)
            tgtsB   = batch.target.float()
            batch.x = batch.x.float()

            out, rel_symbol = model(
                batch.x, batch.edge_index, batch.edge_attr,
                q_idx_batch, new_obj_idx_batch
            )

            # ── object symbols (per node, then index per relation) ────────────
            obj_bits, _ = model.get_object_symbols(batch.x)   # [Total_N, obj_sym]
            rec_query_obj_sym.extend(obj_bits[q_idx_batch].cpu().int().tolist())
            rec_new_obj_sym.extend(obj_bits[new_obj_idx_batch].cpu().int().tolist())

            bce_loss = F.binary_cross_entropy_with_logits(
                out[:, OUT_DIM], tgtsB[:, OUT_DIM],
                pos_weight=pos_weight, reduction='mean')

            mask_intact          = (tgtsB[:, OUT_DIM] < 0.5).float().unsqueeze(1)
            raw_mse              = mse_loss(out[:, :OUT_DIM], tgtsB[:, :OUT_DIM])
            masked_mse           = raw_mse * mask_intact
            sum_mse_per_dim      = masked_mse.sum(dim=0)
            num_valid            = mask_intact.sum()
            mean_mse_per_dim     = sum_mse_per_dim / (num_valid + 1e-8) * W_MSE_DIMS.to(device)
            mse_reg_loss         = mean_mse_per_dim.mean()
            loss                 = W_BCE * bce_loss + W_MSE_REG * mse_reg_loss

            n = len(q_idx_batch)
            total_loss += loss.item() * n
            total_bce  += bce_loss.item() * n
            total_mse  += mse_reg_loss.item() * n
            total_n    += n

            rec_pred.extend(out.cpu().tolist())
            rec_true.extend(tgtsB.cpu().tolist())

            # ── relation symbol ──────────────────────────────────────────────
            if rel_symbol is not None:
                rec_rel_sym.extend(rel_symbol.cpu().tolist())

            if hasattr(batch, 'meta_path'):
                rec_path.extend([x for sub in batch.meta_path for x in sub])
            if hasattr(batch, 'meta_obj_type'):
                rec_obj.extend([x for sub in batch.meta_obj_type for x in sub])
            if hasattr(batch, 'meta_bbox_json'):
                rec_bbox.extend([x for sub in batch.meta_bbox_json for x in sub])
            if hasattr(batch, 'meta_length'):
                rec_len.extend([x for sub in batch.meta_length for x in sub])
            if hasattr(batch, 'meta_q_idx'):
                rec_q.extend(batch.meta_q_idx.cpu().tolist())

    d = total_n or 1
    print(f"\n  Test loss={total_loss/d:.4e}  bce={total_bce/d:.4e}  mse={total_mse/d:.4e}")
    return rec_pred, rec_true, rec_rel_sym, rec_query_obj_sym, rec_new_obj_sym, rec_path, rec_obj, rec_q, rec_len, rec_bbox


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    train_rows, val_rows, test_rows = load_data(TRAIN_VAL_SCRIPT)
    summarise_towers(test_rows, "test")

    raw_weight     = get_positive_weight(train_rows)
    POS_WEIGHT_COL = torch.tensor(raw_weight * 0.5, device=device)

    # Normalisation stats computed on train split (must match training)
    STATS_FOR_NORM = None
    MINMAX = None
    STATS_FOR_NORM = calculate_bbox_stats(train_rows)
    print(STATS_FOR_NORM)

    print("→ Building image bank for test rows…")
    test_imgs = build_bank_optimized(test_rows, UNIQUE_DEPTH_DIR)

    # Build model and load checkpoint
    model, _ = load_ckpt(RUN_NAME, tag="best", device=device)
    print_model_info(model)

    # ── Auto-fit clusters if centroids were not saved yet ─────────────────────
    if model.cluster_centroids.numel() == 0:
        print("\n⚠️  No saved centroids found — fitting clusters now over all splits...")
        all_rows  = train_rows + val_rows + test_rows
        all_imgs  = build_bank_optimized(all_rows, UNIQUE_DEPTH_DIR)
        all_dataset = preprocess_to_dataset_normalized(all_rows, all_imgs, device, STATS_FOR_NORM, MINMAX)
        fit_loader  = DataLoader(all_dataset, batch_size=REL_GRAPHS, shuffle=False, num_workers=0)
        model.fit_clusters(fit_loader)
        save_clusters(model, RUN_NAME, tag="best")
        print("✅ Centroids fitted and saved — future runs will load them automatically.\n")

    print("→ Preprocessing test data…")
    test_dataset = preprocess_to_dataset_normalized(test_rows, test_imgs, device, STATS_FOR_NORM, MINMAX)
    test_loader  = DataLoader(test_dataset, batch_size=REL_GRAPHS, shuffle=False, num_workers=0)

    te_pred, te_true, te_rel_sym, te_query_obj_sym, te_new_obj_sym, te_path, te_obj, te_q, te_len, te_bb = \
        run_test(test_loader, model, POS_WEIGHT_COL)
    
    calculate_and_print_metrics(te_pred, te_true, "TEST (Current Best)", STATS_FOR_NORM, MINMAX)

    te_pred_t, te_true_t = reverse_calculation(te_pred, te_true, STATS_FOR_NORM, MINMAX)
    te_pred_t[:, OUT_DIM] = torch.sigmoid(te_pred_t[:, OUT_DIM])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame({
        "rgb_image_path":     te_path,
        "object_type":        te_obj,
        "query_index":        te_q,
        "length":             te_len,
        "bbox_json":          te_bb,
        "predicted":          te_pred_t.tolist(),
        "actual":             te_true_t.tolist(),
        "symbol":         te_rel_sym,
        "obj_symbol_query":   te_query_obj_sym,
        "obj_symbol_new":     te_new_obj_sym,
    })
    out_path = os.path.join(LOG_DIR, f"results_TEST_seed{SEED}_{timestamp}.xlsx")
    df.to_excel(out_path, index=False)
    print(f"\n✅ Test results → {out_path}")