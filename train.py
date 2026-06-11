import os, time, argparse, shutil
import torch, pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from datetime import datetime
import yaml

from model import TGModel
from load_data import (load_data, get_positive_weight, preprocess_to_dataset_normalized,
                       set_seed, summarise_towers, calculate_bbox_stats, calculate_and_print_metrics, reverse_calculation)
import load_data as _load_data
from utils import print_model_info, build_bank_optimized

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="configs/train.yaml")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

# Populate load_data module globals from yaml
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
EPOCHS            = cfg["training"]["epochs"]
LR                = cfg["training"]["lr"]
FACTOR            = cfg["training"]["lr_factor"]
PATIENCE          = cfg["training"]["lr_patience"]
EARLY_STOP        = cfg["training"]["early_stop_patience"]

NORMALIZATION     = True
MINMAX_SCALING    = False
GLOBAL            = False

# ── Setup ─────────────────────────────────────────────────────────────────────
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_DIR = os.path.join("logs", RUN_NAME)
os.makedirs(LOG_DIR, exist_ok=True)

# Save config snapshot
shutil.copy(args.config, os.path.join(LOG_DIR, "train.yaml"))

mse_loss = nn.MSELoss(reduction='none')

print("Using device:", device)
print(f"Run: {RUN_NAME}  →  {LOG_DIR}")

# ── Model, optimizer, data ────────────────────────────────────────────────────
model = TGModel(2, cfg["model"]["hidden_dim"], symbol_size=SYM_SIZE,
                obj_symbol_size=OBJ_SYM_SIZE, discrete=DISCRETE,
                symbolic=SYMBOLIC).to(device)
model.gs_layer.hard = False
print_model_info(model)

def make_optimizer():
    trainable = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(trainable, lr=LR, weight_decay=1e-4)

optimizer = make_optimizer()
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=FACTOR, patience=PATIENCE, verbose=True, min_lr=1e-6)

train_rows, val_rows, test_rows = load_data(TRAIN_VAL_SCRIPT)
summarise_towers(train_rows, "train")
summarise_towers(val_rows,   "val")
summarise_towers(test_rows,  "test")

raw_weight    = get_positive_weight(train_rows)
POS_WEIGHT_COL = torch.tensor(raw_weight * 0.5, device=device)
print("collapse pos_weight:", POS_WEIGHT_COL)

print("→ Building unique image cache for training …", end=""); t = time.perf_counter()
train_imgs = build_bank_optimized(train_rows, UNIQUE_DEPTH_DIR)
print(f"{time.perf_counter()-t:.1f}s")

print("→ Building unique image cache for validation …", end=""); t = time.perf_counter()
val_imgs = build_bank_optimized(val_rows, UNIQUE_DEPTH_DIR)
print(f"{time.perf_counter()-t:.1f}s")

print("→ Building unique image cache for test …", end=""); t = time.perf_counter()
test_imgs = build_bank_optimized(test_rows, UNIQUE_DEPTH_DIR)
print(f"{time.perf_counter()-t:.1f}s")

STATS_FOR_NORM = None
MINMAX = None

STATS_FOR_NORM = calculate_bbox_stats(train_rows)
print(STATS_FOR_NORM)


# ── run_pass — identical to original ─────────────────────────────────────────
def run_pass(loader, training):
    if training:
        model.train()
        torch.set_grad_enabled(True)
        optimizer.zero_grad()
    else:
        model.eval()
        torch.set_grad_enabled(False)
    total_loss = 0.0
    total_rel  = 0
    total_bce_loss_unweighted      = 0.0
    total_mse_reg_loss_unweighted  = 0.0
    total_mse_reg_dims_unweighted  = torch.zeros(OUT_DIM, device=device)
    rec_pred, rec_true, rec_pred_symbol = [], [], []
    rec_path, rec_obj_type, rec_q, rec_len, rec_bbox = [], [], [], [], []
    for batch in loader:
        given_nodes_list    = []
        new_object_node_list = []
        for i in range(batch.num_graphs):
            node_ids          = (batch.batch == i).nonzero(as_tuple=True)[0]
            nodes_except_last = node_ids[:-1]
            given_nodes_list.append(nodes_except_last)
            node_last = torch.stack([node_ids[-1]] * len(nodes_except_last))
            new_object_node_list.append(node_last)
        if not given_nodes_list: continue
        q_idx_batch       = torch.cat(given_nodes_list)
        new_obj_idx_batch = torch.cat(new_object_node_list)
        tgtsB  = batch.target.float()
        batch.x = batch.x.float()
        out, symbol = model(batch.x, batch.edge_index, batch.edge_attr,
                            q_idx_batch, new_obj_idx_batch)
        pred_flag = out[:, OUT_DIM]
        tgt_flag  = tgtsB[:, OUT_DIM]
        bce_loss  = F.binary_cross_entropy_with_logits(
            pred_flag, tgt_flag, pos_weight=POS_WEIGHT_COL, reduction='mean')
        pred_bbox    = out[:, :OUT_DIM]
        tgt_bbox     = tgtsB[:, :OUT_DIM]
        mask_intact  = (tgt_flag < 0.5).float().unsqueeze(1)
        raw_mse      = mse_loss(pred_bbox, tgt_bbox)
        masked_mse   = raw_mse * mask_intact
        sum_mse_per_dim  = masked_mse.sum(dim=0)
        num_valid_samples = mask_intact.sum()
        mean_mse_per_dim  = sum_mse_per_dim / (num_valid_samples + 1e-8) * W_MSE_DIMS.to(device)
        mse_reg_loss      = mean_mse_per_dim.mean()
        loss = (W_BCE * bce_loss) + (W_MSE_REG * mse_reg_loss)
        if training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        num_rels_in_batch = len(q_idx_batch)
        total_loss                    += loss.item()       * num_rels_in_batch
        total_rel                     += num_rels_in_batch
        total_bce_loss_unweighted     += bce_loss.item()  * num_rels_in_batch
        total_mse_reg_loss_unweighted += mse_reg_loss.item() * num_rels_in_batch
        total_mse_reg_dims_unweighted += mean_mse_per_dim.detach() * num_rels_in_batch
        rec_pred += out.detach().cpu().tolist()
        rec_true += tgtsB.detach().cpu().tolist()
        if not training:
            if symbol is not None:
                rec_pred_symbol += symbol.detach().cpu().tolist()
            if hasattr(batch, 'meta_path'):
                rec_path.extend([item for sublist in batch.meta_path for item in sublist])
            if hasattr(batch, 'meta_obj_type'):
                rec_obj_type.extend([item for sublist in batch.meta_obj_type for item in sublist])
            if hasattr(batch, 'meta_bbox_json'):
                rec_bbox.extend([item for sublist in batch.meta_bbox_json for item in sublist])
            if hasattr(batch, 'meta_length'):
                rec_len.extend([item for sublist in batch.meta_length for item in sublist])
            if hasattr(batch, 'meta_q_idx'):
                rec_q.extend(batch.meta_q_idx.cpu().tolist())
    avg              = total_loss / total_rel if total_rel else 0.0
    avg_bce_loss     = total_bce_loss_unweighted     / total_rel if total_rel else 0.0
    avg_mse_reg_loss = total_mse_reg_loss_unweighted / total_rel if total_rel else 0.0
    avg_mse_reg_dims = total_mse_reg_dims_unweighted / total_rel if total_rel else torch.zeros(OUT_DIM, device=device)
    return (avg, avg_bce_loss, avg_mse_reg_loss, avg_mse_reg_dims,
            rec_path, rec_obj_type, rec_q, rec_len, rec_bbox, rec_pred, rec_pred_symbol, rec_true)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("→ Preprocessing training data into DataLoader...")
    train_dataset = preprocess_to_dataset_normalized(train_rows, train_imgs, device, STATS_FOR_NORM, MINMAX)
    train_loader  = DataLoader(train_dataset, batch_size=REL_GRAPHS, shuffle=True,  num_workers=0)

    print("→ Preprocessing validation data into DataLoader...")
    val_dataset = preprocess_to_dataset_normalized(val_rows, val_imgs, device, STATS_FOR_NORM, MINMAX)
    val_loader  = DataLoader(val_dataset, batch_size=REL_GRAPHS, shuffle=True, num_workers=0)

    best_val     = float("inf")
    no_imp       = 0
    start_epoch  = 0
    save_filename = None
    best_ckpt_path = None

    # ── Resume ────────────────────────────────────────────────────────────────
    LAST_CKPT = os.path.join(LOG_DIR, "last.pt")
    if args.resume and os.path.isfile(LAST_CKPT):
        print(f"→ Resuming from {LAST_CKPT}")
        checkpoint = torch.load(LAST_CKPT, map_location=device, weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            try:
                if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except ValueError:
                optimizer = make_optimizer()
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=FACTOR, patience=PATIENCE, verbose=True, min_lr=1e-6)
            if 'best_val_loss' in checkpoint: best_val = checkpoint['best_val_loss']
            if 'epoch'         in checkpoint: start_epoch = checkpoint['epoch']
        except (KeyError, TypeError):
            model.load_state_dict(checkpoint)
        print(f"   Resumed at epoch {start_epoch}, best_val={best_val:.4e}")
    elif args.resume:
        print(f"⚠️  --resume given but {LAST_CKPT} not found — starting fresh.")

    train_hist, val_hist = [], []
    best_train_cache, best_val_cache = None, None

    for ep in range(start_epoch, EPOCHS):
        tr_loss, tr_bce, tr_mse_reg, tr_mse_dims, t_path, t_obj, t_q, t_len, t_bb, t_pred, t_symbol, t_true = run_pass(train_loader, True)
        with torch.no_grad():
            val_loss, val_bce, val_mse_reg, val_mse_dims, v_path, v_obj, v_q, v_len, v_bb, v_pred, v_symbol, v_true = run_pass(val_loader, False)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        if ep % 100 == 0 or val_loss < best_val:
            print(f"[{ep+1:05d}] lr={lr:.1e} {'── NEW BEST' if val_loss < best_val else ''}")
            print(f"    TR (total={tr_loss:.4e}) | collapse={tr_bce:.4e} | mse_reg={tr_mse_reg:.4e}")
            print(f"    VAL(total={val_loss:.4e}) | collapse={val_bce:.4e} | mse_reg={val_mse_reg:.4e}")

        train_hist.append(tr_loss); val_hist.append(val_loss)

        # Always save last.pt
        torch.save({'epoch': ep+1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val},
                   os.path.join(LOG_DIR, "last.pt"))

        if val_loss < best_val:
            best_val = val_loss; no_imp = 0
            # Delete previous best, save new one
            new_save_filename = f"best_epoch_{ep+1}_val_{val_loss:.4e}.pt"
            new_save_path     = os.path.join(LOG_DIR, new_save_filename)
            torch.save({'epoch': ep+1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val}, new_save_path)
            if best_ckpt_path and os.path.isfile(best_ckpt_path):
                os.remove(best_ckpt_path)
                print(f"   🗑  Removed: {os.path.basename(best_ckpt_path)}")
            best_ckpt_path = new_save_path
            save_filename  = new_save_filename
            print(f"   🟢 best model saved to {new_save_path}")
            if ep > 0:
                best_train_cache = (t_path, t_obj, t_q, t_len, t_bb, t_pred, t_symbol, t_true)
                best_val_cache   = (v_path, v_obj, v_q, v_len, v_bb, v_pred, v_symbol, v_true)
                print("   " + "─" * 70)
                print(f"   ✨ NEW BEST METRICS (Epoch {ep+1}) ✨")
                calculate_and_print_metrics(t_pred, t_true, "Train (Current Best)",      STATS_FOR_NORM, MINMAX)
                calculate_and_print_metrics(v_pred, v_true, "Validation (Current Best)", STATS_FOR_NORM, MINMAX)
                print("   " + "─" * 70)
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP:
                print("🛑 early stop"); break

    if best_val_cache is None:
        with torch.no_grad():
            val_loss, val_bce, val_mse_reg, val_mse_dims, v_path, v_obj, v_q, v_len, v_bb, v_pred, v_symbol, v_true = run_pass(val_loader, False)
        best_val_cache = (v_path, v_obj, v_q, v_len, v_bb, v_pred, v_symbol, v_true)

    v_path, v_obj, v_q, v_len, v_bb, v_pred, v_symbol, v_true = best_val_cache
    print(f"✅ Using (val = {best_val:.4e}) for reports")

    print("\n--- DATAFRAME LENGTH CHECK (VAL) ---")
    print(f"rgb_image_path : {len(v_path)}")
    print(f"object_type    : {len(v_obj)}")
    print(f"query_index    : {len(v_q)}")
    print(f"length         : {len(v_len)}")
    print(f"bbox_json      : {len(v_bb)}")
    print(f"predicted      : {len(v_pred)}")
    print(f"actual         : {len(v_true)}")
    print(f"symbol         : {len(v_symbol) if v_symbol else 0}")
    print("------------------------------------\n")

    v_pred_t, v_true_t = reverse_calculation(v_pred, v_true, STATS_FOR_NORM, MINMAX)
    v_pred_t[:, OUT_DIM] = torch.sigmoid(v_pred_t[:, OUT_DIM])
    df_val = pd.DataFrame({
        "rgb_image_path": v_path,
        "object_type":    v_obj,
        "query_index":    v_q,
        "length":         v_len,
        "bbox_json":      v_bb,
        "predicted":      v_pred_t.tolist(),
        "actual":         v_true_t.tolist(),
        "symbol":         v_symbol,
    })
    calculate_and_print_metrics(v_pred, v_true, "Validation (Current Best)", STATS_FOR_NORM, MINMAX)
    timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = os.path.join(LOG_DIR, f"results_val_{timestamp}.xlsx")
    df_val.to_excel(results_filename, index=False)
    print(f"Validation results saved to {results_filename}")

    best_model_path = os.path.join(LOG_DIR, save_filename) if save_filename else None
    if best_model_path and os.path.exists(best_model_path):
        print(f"\n🔄 Reloading BEST weights from: {save_filename} for Test Pass...")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        print(f"\n⚠️ Warning: Best model file not found. Using current weights.")

    print("\n" + "="*50)
    print("       RUNNING FINAL TEST PASS (BEST WEIGHTS)")
    print("="*50)

    print("→ Preprocessing TEST data into DataLoader...")
    test_dataset = preprocess_to_dataset_normalized(test_rows, test_imgs, device, STATS_FOR_NORM, MINMAX)
    test_loader  = DataLoader(test_dataset, batch_size=REL_GRAPHS, shuffle=False, num_workers=0)

    with torch.no_grad():
        te_loss, te_bce, te_mse_reg, te_mse_dims, te_path, te_obj, te_q, te_len, te_bb, te_pred, te_symbol, te_true = run_pass(test_loader, False)

    te_pred_t, te_true_t = reverse_calculation(te_pred, te_true, STATS_FOR_NORM, MINMAX)
    te_pred_t[:, OUT_DIM] = torch.sigmoid(te_pred_t[:, OUT_DIM])
    df_test = pd.DataFrame({
        "rgb_image_path": te_path,
        "object_type":    te_obj,
        "query_index":    te_q,
        "length":         te_len,
        "bbox_json":      te_bb,
        "predicted":      te_pred_t.tolist(),
        "actual":         te_true_t.tolist(),
        "symbol":         te_symbol,
    })
    calculate_and_print_metrics(te_pred, te_true, "TEST (Current Best)", STATS_FOR_NORM, MINMAX)
    test_results_filename = os.path.join(LOG_DIR, f"results_TEST_seed{SEED}_{timestamp}.xlsx")
    df_test.to_excel(test_results_filename, index=False)
    print(f"✅ FINAL TEST results (Best Weights) saved to {test_results_filename}")