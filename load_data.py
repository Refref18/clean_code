import os, csv, ast, json, random
import torch, matplotlib.pyplot as plt, pandas as pd
import numpy as np
import torch.nn.functional as F
from glob import glob
from collections import defaultdict, Counter
import torch.nn.functional as F
import math
import ast
from utils import update_graph, finalize_graph_with_image

# ── Config (populated once by train.py calling init(cfg)) ────────────────────
device       = None
SEED         = 42
MAX_ROWS     = 50_000
VAL_RATIO    = 0.2
TEST_RATIO   = 0.2
SCALE        = 100
OUT_DIM      = 4
BB_DIM       = 4
TOWER_HEIGHT = 4
ALPHA_VALS   = [1.0, 1.0, 1.0, 1.0]
THRESH_COLLAPSE = 0.5

def init(cfg: dict):
    """Called once from train.py to populate config from yaml."""
    global device, SEED, MAX_ROWS, VAL_RATIO, TEST_RATIO
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED         = cfg["seed"]
    MAX_ROWS     = cfg["data"]["max_rows"]
    VAL_RATIO    = cfg["data"]["val_ratio"]
    TEST_RATIO   = cfg["data"]["test_ratio"]

# ── Unchanged helpers ─────────────────────────────────────────────────────────
def set_seed(SEED):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    def set_seed(seed: int):
        random.seed(seed)
        import numpy as _np
        _np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    set_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def get_alpha_tensor(target_device):
    return torch.tensor(ALPHA_VALS, device=target_device).view(1, -1)

def symlog(x):
    alpha = get_alpha_tensor(x.device)
    return torch.sign(x) * torch.log1p(x.abs() * alpha)

def inv_symlog(y):
    alpha = get_alpha_tensor(y.device)
    return torch.sign(y) * (torch.expm1(y.abs()) / alpha)

def _update_collapse_stats(stats, logits, targets, thresh=0.5):
    pred = (torch.sigmoid(logits) >= thresh)
    targ = (targets >= 0.5)
    tp = (pred &  targ).sum().item()
    tn = (~pred & ~targ).sum().item()
    fp = (pred & ~targ).sum().item()
    fn = (~pred &  targ).sum().item()
    stats["tp"] += tp; stats["tn"] += tn; stats["fp"] += fp; stats["fn"] += fn

def _collapse_metrics_from_stats(stats):
    tp, tn, fp, fn = stats["tp"], stats["tn"], stats["fp"], stats["fn"]
    P, N = tp + fn, tn + fp
    eps  = 1e-12
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "ACC":  (tp + tn) / (P + N + eps),
        "TPR":  tp / (P + eps),
        "TNR":  tn / (N + eps),
        "FPR":  fp / (N + eps),
        "FNR":  fn / (P + eps),
        "PREC": tp / (tp + fp + eps),
        "F1":   2 * tp / (2*tp + fp + fn + eps),
    }

def reverse_calculation(predicted, actual, bbox_stats=None, minmax=None):
    predicted_t = torch.tensor(predicted)
    actual_t    = torch.tensor(actual)
    if bbox_stats is not None:
        target_device = predicted_t.device
        order      = ['MinX', 'MinZ', 'MaxX', 'MaxZ']
        means_list = [bbox_stats[k]['mean'] for k in order]
        stds_list  = [bbox_stats[k]['std']  for k in order]
        stat_mean  = torch.tensor(means_list, device=target_device).view(1, 4)
        stat_std   = torch.tensor(stds_list,  device=target_device).view(1, 4)
        predicted_t[:, :OUT_DIM] = (predicted_t[:, :OUT_DIM] * stat_std) + stat_mean
        actual_t[:, :OUT_DIM]    = (actual_t[:, :OUT_DIM]    * stat_std) + stat_mean
    predicted_t[:, :OUT_DIM] = inv_symlog(predicted_t[:, :OUT_DIM]) / SCALE
    actual_t[:, :OUT_DIM]    = inv_symlog(actual_t[:, :OUT_DIM])    / SCALE
    return predicted_t, actual_t

def calculate_and_print_metrics(pred_list, true_list, dataset_name, bbox_stats=None, minmax=None):
    print(f"\n  Metrics for: {dataset_name.upper()}")
    if not pred_list or not true_list:
        print("    No data to report.")
        return
    with torch.no_grad():
        pred = torch.tensor(pred_list)
        true = torch.tensor(true_list)
        pred_bbox          = pred[:, :OUT_DIM]
        true_bbox          = true[:, :OUT_DIM]
        true_collapse_flag = true[:, OUT_DIM]
        mask_intact        = (true_collapse_flag < 0.5)
        print("    BBox MSE (on intact samples) [REAL WORLD UNITS]:")
        if mask_intact.sum() > 0:
            pred_bbox_intact = pred_bbox[mask_intact].clone()
            true_bbox_intact = true_bbox[mask_intact].clone()
            print(f"      Original losses")
            per_dim_mse    = F.mse_loss(pred_bbox_intact, true_bbox_intact, reduction='none').mean(dim=0)
            dim_labels_mse = ["Dim 0 (min_x)", "Dim 1 (min_z)", "Dim 2 (max_x)", "Dim 3 (max_z)"]
            for i in range(OUT_DIM):
                print(f"      {dim_labels_mse[i]}: MSE={per_dim_mse[i].item():.4e}")
            if bbox_stats is not None:
                target_device    = pred_bbox_intact.device
                order            = ['MinX', 'MinZ', 'MaxX', 'MaxZ']
                stat_mean        = torch.tensor([bbox_stats[k]['mean'] for k in order], device=target_device).view(1, 4)
                stat_std         = torch.tensor([bbox_stats[k]['std']  for k in order], device=target_device).view(1, 4)
                pred_bbox_intact = (pred_bbox_intact * stat_std) + stat_mean
                true_bbox_intact = (true_bbox_intact * stat_std) + stat_mean
            pred_bbox_intact_rev = inv_symlog(pred_bbox_intact) / SCALE
            true_bbox_intact_rev = inv_symlog(true_bbox_intact) / SCALE
            mse_val = F.mse_loss(pred_bbox_intact_rev, true_bbox_intact_rev)
            print(f"      Overall MSE (m²): {mse_val.item():.4e}")
            per_dim_mse = F.mse_loss(pred_bbox_intact_rev, true_bbox_intact_rev, reduction='none').mean(dim=0)
            for i in range(OUT_DIM):
                rmse_cm = math.sqrt(per_dim_mse[i].item()) * 100
                print(f"      {dim_labels_mse[i]}: MSE={per_dim_mse[i].item():.4e} | RMSE={rmse_cm:.4f} cm")
        else:
            print("      No intact samples for MSE metric calculation.")
        pred_collapse_logits = pred[:, OUT_DIM]
        true_collapse_labels = true[:, OUT_DIM]
        stats = defaultdict(int)
        _update_collapse_stats(stats, pred_collapse_logits, true_collapse_labels, thresh=THRESH_COLLAPSE)
        metrics = _collapse_metrics_from_stats(stats)
        print("\n    Collapse Prediction Metrics:")
        print(f"      TP Rate (TPR): {metrics['TPR']:.4f} | TN Rate (TNR): {metrics['TNR']:.4f}")
        print(f"      Accuracy: {metrics['ACC']:.4f} | F1-Score: {metrics['F1']:.4f}")

def _load_rows_from_glob(glob_pat: str, noncollapse_quota: int, seed: int = SEED):
    files = sorted(glob(glob_pat))
    by_tid = defaultdict(list)
    for _p in files:
        with open(_p, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                by_tid[row["id"]].append(row)
    tids = list(by_tid.keys())
    rng = random.Random(seed)
    rng.shuffle(tids)
    rows_shuffled = []
    current_non_collapse_count = 0
    for tid in tids:
        tower_rows = sorted(by_tid[tid], key=lambda r: int(r.get("step", 0) or 0))
        tower_non_collapses = 0
        for r in tower_rows:
            gc_str = r.get('global_collapse', r.get('collapse', 'false'))
            if str(gc_str).lower() != 'true':
                tower_non_collapses += 1
        rows_shuffled.extend(tower_rows)
        current_non_collapse_count += tower_non_collapses
        if current_non_collapse_count >= noncollapse_quota:
            break
    nc = sum(1 for r in rows_shuffled if str(r.get('global_collapse', r.get('collapse', ''))).lower() != 'true')
    cc = len(rows_shuffled) - nc
    print(f"[LOAD] {glob_pat} -> rows={len(rows_shuffled)} | non-collapsed={nc} | collapsed={cc} (Quota: {noncollapse_quota})")
    return rows_shuffled

def load_data(TRAIN_VAL_SCRIPT):
    print(f"--> SINGLE SOURCE MODE: Loading from {TRAIN_VAL_SCRIPT}")
    if MAX_ROWS == -1:
        target_quota = 10_000_000
    else:
        target_quota = int(MAX_ROWS * (1 + TEST_RATIO))
        print(f"--> Loading Target Quota: {target_quota} (includes {MAX_ROWS} for Train/Val + 10% Test)")
    all_rows = _load_rows_from_glob(TRAIN_VAL_SCRIPT, noncollapse_quota=target_quota, seed=SEED)
    by_tid = defaultdict(list)
    for r in all_rows:
        by_tid[r["id"]].append(r)
    all_tids = list(by_tid.keys())
    total_towers = len(all_tids)
    split_idx_test = int(total_towers * (1.0 / (1 + TEST_RATIO)))
    train_val_tids = all_tids[:split_idx_test]
    test_tids = all_tids[split_idx_test:]
    val_count_tids = int(len(train_val_tids) * VAL_RATIO)
    val_tids = train_val_tids[:val_count_tids]
    train_tids = train_val_tids[val_count_tids:]
    def get_sorted_rows_for_tids(tids_list, lookup_dict):
        out = []
        for tid in tids_list:
            t_rows = sorted(lookup_dict[tid], key=lambda r: int(r.get("step", 0)))
            out.extend(t_rows)
        return out
    train_rows = get_sorted_rows_for_tids(train_tids, by_tid)
    val_rows   = get_sorted_rows_for_tids(val_tids,   by_tid)
    test_rows  = get_sorted_rows_for_tids(test_tids,  by_tid)
    print(f"[SPLIT RESULT] Towers -> Train: {len(train_tids)} | Val: {len(val_tids)} | Test: {len(test_tids)}")
    print(f"[SPLIT RESULT] Rows   -> Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")
    print("-" * 30)
    return train_rows, val_rows, test_rows

def load_data_pddl(TRAIN_VAL_SCRIPT):
    print(f"--> SINGLE SOURCE MODE: Loading from {TRAIN_VAL_SCRIPT}")
    if MAX_ROWS == -1:
        target_quota = 10_000_000
    else:
        target_quota = int(MAX_ROWS * (1 + TEST_RATIO))
        print(f"--> Loading Target Quota: {target_quota} (includes {MAX_ROWS} for Train/Val + 10% Test)")
    all_rows = _load_rows_from_glob(TRAIN_VAL_SCRIPT, noncollapse_quota=target_quota, seed=SEED)
    by_tid = defaultdict(list)
    for r in all_rows:
        by_tid[r["id"]].append(r)
    all_tids = list(by_tid.keys())
    total_towers = len(all_tids)
    split_idx_test = int(total_towers * (1.0 / (1 + TEST_RATIO)))
    train_val_tids = all_tids[:split_idx_test]
    val_count_tids = int(len(train_val_tids) * VAL_RATIO)
    val_tids = train_val_tids[:val_count_tids] + all_tids[split_idx_test:]
    train_tids = train_val_tids[val_count_tids:]
    def get_sorted_rows_for_tids(tids_list, lookup_dict):
        out = []
        for tid in tids_list:
            t_rows = sorted(lookup_dict[tid], key=lambda r: int(r.get("step", 0)))
            out.extend(t_rows)
        return out
    train_rows = get_sorted_rows_for_tids(train_tids, by_tid)
    val_rows   = get_sorted_rows_for_tids(val_tids,   by_tid)
    return train_rows, val_rows

def get_positive_weight(train_rows):
    col_pos = 0; col_neg = 0
    for r in train_rows:
        try:
            d = ast.literal_eval(r.get("pairwise_collapse", "{}"))
            p = sum(1 for v in d.values() if v)
            n = len(d) - p
            col_pos += p
            col_neg += n
        except:
            pass
    return max(1.0, col_neg / (col_pos + 1e-6))

def count_by_step(rows, name):
    counts = {}
    total_rows = len(rows)
    collapses_per_step = {}
    for r in rows:
        s = int(r["step"])
        counts[s] = counts.get(s, 0) + 1
        is_collapsed = str(r.get("collapse", "false")).lower() == "true"
        if is_collapsed:
            collapses_per_step[s] = collapses_per_step.get(s, 0) + 1
    print(f"\n── {name.upper()} STEP DISTRIBUTION ──────────────────────")
    print(f"{'Step':<6} | {'Count':<8} | {'% of Total':<12} | {'Collapse Rate':<15}")
    print("─" * 50)
    for s in sorted(counts.keys()):
        count = counts[s]
        pct = (count / total_rows) * 100 if total_rows else 0
        col_count = collapses_per_step.get(s, 0)
        col_rate = (col_count / count) * 100 if count else 0
        print(f"{s:<6} | {count:<8} | {pct:>10.1f}% | {col_rate:>13.1f}%")
    total_collapses = sum(collapses_per_step.values())
    print("─" * 50)
    print(f"TOTAL  | {total_rows:<8} | 100.0%       | {(total_collapses/total_rows)*100:>13.1f}%")
    print("──────────────────────────────────────────────────────")

def summarise_rows(rows, name="set"):
    def run_row_stats(row_list):
        total = len(row_list)
        collapsed = sum(str(r.get("collapse", "false")).lower() == "true" for r in row_list)
        intact = total - collapsed
        pct_col = 100 * collapsed / total if total else 0
        pct_ok  = 100 * intact   / total if total else 0
        return total, collapsed, pct_col, intact, pct_ok
    res_all   = run_row_stats(rows)
    rows_no_s1 = [r for r in rows if int(r.get("step", 0)) != 1]
    res_no_s1 = run_row_stats(rows_no_s1)
    print(f"\n──  {name.upper()} ROW STATISTICS (COMPARATIVE) ───────────────────")
    print(f" [ALL ROWS (Including Step 1)]")
    print(f" Total rows : {res_all[0]:6d} | collapsed {res_all[1]:6d} ({res_all[2]:5.1f}%) | intact {res_all[3]:6d} ({res_all[4]:5.1f}%)")
    print(f"\n [BUILDING ROWS (Excluding Step 1)]")
    print(f" Total rows : {res_no_s1[0]:6d} | collapsed {res_no_s1[1]:6d} ({res_no_s1[2]:5.1f}%) | intact {res_no_s1[3]:6d} ({res_no_s1[4]:5.1f}%)")
    ratio_diff = res_no_s1[2] - res_all[2]
    print(f"\n COLLAPSE DENSITY SHIFT: {res_all[2]:.1f}% -> {res_no_s1[2]:.1f}% (+{ratio_diff:.1f}%)")
    print("─────────────────────────────────────────────────────────────")

def summarise_towers(rows, name="set"):
    towers = {}
    for r in rows:
        tid  = r["id"]
        step = int(r["step"])
        coll = str(r["collapse"]).lower() == "true"
        if tid not in towers:
            towers[tid] = {"max_step": step, "collapsed": coll}
        else:
            towers[tid]["max_step"]   = max(towers[tid]["max_step"], step)
            towers[tid]["collapsed"]  = towers[tid]["collapsed"] or coll
    collapsed = [t for t in towers.values() if t["collapsed"]]
    intact    = [t for t in towers.values() if not t["collapsed"]]
    n_tot, n_col, n_ok = len(towers), len(collapsed), len(intact)
    pct_col = 100 * n_col / n_tot if n_tot else 0
    pct_ok  = 100 * n_ok  / n_tot if n_tot else 0
    avg_h_col = np.mean([t["max_step"] for t in collapsed]) if n_col else 0
    avg_h_ok  = np.mean([t["max_step"] for t in intact])    if n_ok  else 0
    print(f"\n──  {name.upper()} STATISTICS ──────────────────────────")
    print(f" towers total : {n_tot}")
    print(f"   collapsed  : {n_col:5d}  ({pct_col:5.1f} %)   avg height = {avg_h_col:.2f}  steps")
    print(f"   intact     : {n_ok:5d}  ({pct_ok :5.1f} %)   avg height = {avg_h_ok :.2f}  steps")
    print("───────────────────────────────────────────────────────────")

def calculate_bbox_stats(rows, save_path="bbox_distribution.png"):
    processed_values = []
    for row in rows:
        if str(row.get("step")) == "1": continue
        bb_str = row.get("bounding_box_differences")
        if not bb_str or bb_str == "None": continue
        bb = ast.literal_eval(bb_str)
        pw_str = row.get("pairwise_collapse", "{}")
        pairwise_dict = ast.literal_eval(pw_str)
        for i, entry in bb.items():
            is_collapsed = pairwise_dict.get(i, False) or pairwise_dict.get(str(i), False)
            if not is_collapsed:
                try:
                    vals = [entry['min_diff'][0], entry['min_diff'][2],
                            entry['max_diff'][0], entry['max_diff'][2]]
                    processed_values.append(vals)
                except (KeyError, TypeError, IndexError): continue
    if not processed_values:
        print("No valid data found after filtering.")
        return None
    raw_tensor         = torch.tensor(processed_values, dtype=torch.float)
    scaled_tensor      = raw_tensor * SCALE
    transformed_tensor = symlog(scaled_tensor)
    data_np = transformed_tensor.numpy()
    means   = np.mean(data_np, axis=0)
    stds    = np.std(data_np,  axis=0)
    labels  = ["MinX", "MinZ", "MaxX", "MaxZ"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"Transformed BBox Distributions (Scale: {SCALE}, Symlog applied)", fontsize=14)
    for i in range(4):
        axes[i].hist(data_np[:, i], bins=40, color='mediumseagreen', edgecolor='black', alpha=0.7)
        axes[i].axvline(means[i], color='red', linestyle='dashed', linewidth=2, label=f'Mean: {means[i]:.4f}')
        axes[i].set_title(f"{labels[i]}")
        axes[i].set_xlabel("Symlog Transformed Value")
        axes[i].legend()
        stats_text = f"Mean: {means[i]:.4f}\nStd: {stds[i]:.4f}"
        axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    return {labels[i]: {"mean": means[i], "std": stds[i]} for i in range(4)}


def preprocess_to_dataset_normalized(rows, imgs_data, device, bbox_stats=None, minmax=None):
    """
    Identical to original. imgs_data is a tuple (image_cache, match_registry).
    """
    dataset_list = []
    if isinstance(imgs_data, tuple) and len(imgs_data) == 2:
        image_cache, match_registry = imgs_data
        def get_image_for_row(row):
            obj_type     = row.get('object_type')
            obj_size_raw = row.get('object_size_or_path')
            lookup_key   = (obj_type, obj_size_raw)
            if lookup_key in image_cache:
                return image_cache[lookup_key]
            else:
                print(f"⚠️ Warning: Image not found for {lookup_key}")
                return torch.zeros((2, 64, 64), device=device)
        sample_img = next(iter(image_cache.values()))
    empty_img = torch.zeros_like(sample_img).to(device)
    current_graph, prev_act = None, 0.0
    if bbox_stats is not None:
        order      = ['MinX', 'MinZ', 'MaxX', 'MaxZ']
        means_list = [bbox_stats[k]['mean'] for k in order]
        stds_list  = [bbox_stats[k]['std']  for k in order]
        stat_mean  = torch.tensor(means_list, device=device).view(1, 4)
        stat_std   = torch.tensor(stds_list,  device=device).view(1, 4)
    for idx, row in enumerate(rows):
        global_coll  = str(row.get("collapse", "false")).lower() == "true"
        pairwise_dict = ast.literal_eval(row.get('pairwise_collapse', '{}'))
        obj_type     = row.get('object_type')
        obj_size     = row.get('object_size_or_path')
        img          = get_image_for_row(row).to(device)
        if row["step"] == "1":
            current_graph = update_graph(current_graph, img, {}, BB_DIM)
            continue
        else:
            bb           = ast.literal_eval(row.get("bounding_box_differences"))
            spatial_dict = ast.literal_eval(row.get('spatial_relations', '{}'))
            rel_keys     = ["below", "on_top", "surrounded", "is_surrounding",
                            "inside_full", "inside_50", "inside_20", "occluding"]
        path    = row.get('rgb_image_path', 'NA')
        if current_graph:
            N           = len(current_graph.x)
            data_sample = finalize_graph_with_image(current_graph.clone(), BB_DIM, img).to(device)
            if N >= 1:
                pairwise_labels = []
                bbox_target_list = []
                for i in range(0, N):
                    is_collapsed = pairwise_dict.get(i, False) or pairwise_dict.get(str(i), False)
                    pairwise_labels.append(1.0 if is_collapsed else 0.0)
                    if not is_collapsed and bb:
                        try:
                            entry = bb.get(i, bb.get(str(i)))
                            if entry:
                                vals = [entry['min_diff'][0], entry['min_diff'][2],
                                        entry['max_diff'][0], entry['max_diff'][2]]
                                bbox_target_list.append(vals)
                            else:
                                bbox_target_list.append([0.0] * OUT_DIM)
                        except (KeyError, IndexError, TypeError):
                            bbox_target_list.append([0.0] * OUT_DIM)
                    else:
                        bbox_target_list.append([0.0] * OUT_DIM)
                raw_bbox_target = torch.tensor(bbox_target_list, dtype=torch.float, device=device) * SCALE
                bbox_target     = symlog(raw_bbox_target)
                if bbox_stats is not None:
                    bbox_target = (bbox_target - stat_mean) / stat_std
                collapse_target     = torch.tensor(pairwise_labels, device=device, dtype=torch.float).unsqueeze(1)
                data_sample.target  = torch.cat([bbox_target, collapse_target], dim=1)
                data_sample.meta_path      = [path] * N
                data_sample.meta_obj_type  = [(obj_type, obj_size)] * N
                data_sample.meta_length    = [N] * N
                bb_list_of_dicts           = [bb.get(i, {}) for i in range(0, N)]
                data_sample.meta_bbox_json = [json.dumps(b) for b in bb_list_of_dicts]
                spatial_vectors = []
                for i in range(N):
                    vec = [bool(spatial_dict.get(k, {}).get(i, spatial_dict.get(k, {}).get(str(i), False))) for k in rel_keys]
                    spatial_vectors.append(vec)
                data_sample.meta_spatial_json = [json.dumps(v) for v in spatial_vectors]
                data_sample.meta_q_idx        = torch.arange(N, dtype=torch.long)
                data_sample.id             = row.get('id', 'NA')
                data_sample.step           = row.get('step', 'NA')
                data_sample.tower_collapse = row.get('collapse', 'false')
                dataset_list.append(data_sample)
        current_graph = update_graph(current_graph, img, bb, BB_DIM)
        if current_graph:
            current_graph.x          = current_graph.x.to(device)
            current_graph.edge_index = current_graph.edge_index.to(device)
            if current_graph.edge_attr is not None:
                current_graph.edge_attr = current_graph.edge_attr.to(device)
        if current_graph and (len(current_graph.x) >= TOWER_HEIGHT or global_coll):
            current_graph = None
    return dataset_list