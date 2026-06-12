import pandas as pd
import os
import ast
import sys
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

# =================CONFIGURATION=================
# 1. PATHS
results_file_path = "logs/symcan_trial3_2/results_TEST_seed42_20260611_182955.xlsx"
dataset_file_path = "data.csv" #"data_balanced_clean_4mm.csv"#"data/recreated_all.csv"
base_image_path = "/home/color/Masaüstü/SIU-SymMGN/"
output_root_dir   = './RESULT_last'
plots_subfolder_name = '_PLOTS_SUMMARY'

import shutil

if os.path.exists(output_root_dir):
    shutil.rmtree(output_root_dir)
os.makedirs(output_root_dir, exist_ok=True)

OUT_DIM = 4

# 2. LOSS CALCULATION CONSTANTS
POS_WEIGHT_COL = 10.0

# 3. TOGGLES
SAVE_STITCHED_IMAGES = False
# ===============================================


# ─────────────────────────────────────────────────────────────────────────────
# SITUATION RECOMPUTATION (same fix as plotting + balancing scripts)
# ─────────────────────────────────────────────────────────────────────────────
def _determine_situation_recomputed(aabb_A, aabb_B, eps=0):
    """
    aabb_A = last/newest object  →  dict with 'min':[x,y,z], 'max':[x,y,z]
    aabb_B = older object        →  dict with 'min':[x,y,z], 'max':[x,y,z]

    Key fix: 'occlude' requires A's bottom to be BELOW B's top (z_bot_A < z_top_B).
    A wider object merely sitting on top is labelled 'on_top', not 'occlude'.
    """
    ax_min, ax_max = aabb_A['min'][0], aabb_A['max'][0]
    az_min, az_max = aabb_A['min'][2], aabb_A['max'][2]

    bx_min, bx_max = aabb_B['min'][0], aabb_B['max'][0]
    bz_min, bz_max = aabb_B['min'][2], aabb_B['max'][2]

    z_top_A, z_bot_A = az_max, az_min
    z_top_B          = bz_max

    area_A = (ax_max - ax_min) * (az_max - az_min)
    area_B = (bx_max - bx_min) * (bz_max - bz_min)

    inter_x    = max(0, min(ax_max, bx_max) - max(ax_min, bx_min))
    inter_z    = max(0, min(az_max, bz_max) - max(az_min, bz_min))
    inter_area = inter_x * inter_z

    # A contains B in X → potential occlude
    a_contains_b_x = (ax_min <= bx_min + eps) and (ax_max >= bx_max - eps)
    if a_contains_b_x:
        coverage_of_B = inter_area / area_B if area_B > 0 else 0
        if coverage_of_B > 0.60:
            if z_bot_A < z_top_B:   # THE FIX: must physically overlap in Z
                return "occlude"
            else:
                return "on_top"

    # B contains A in X → inside
    b_contains_a_x = (ax_min >= bx_min - eps) and (ax_max <= bx_max + eps)
    coverage_of_A  = inter_area / area_A if area_A > 0 else 0

    if b_contains_a_x:
        if coverage_of_A > 0.60:
            return "inside"
    elif coverage_of_A > 0.20 and z_top_A - z_top_B > 0.03:
        return "bit-inside"

    if z_top_A - z_top_B > 0.03 and z_bot_A - bz_min > 0.03:
        return "on_top"

    return "unknown"


def _get_situation(bbox_diff_dict, all_aabbs, query_idx, step):
    """
    Single helper used everywhere situation is needed.
    Tries to recompute from raw bbox; falls back to stored label.
    
    bbox_diff_dict : parsed bounding_box_differences dict  (keys may be int or str)
    all_aabbs      : parsed bbox column dict (integer-keyed)
    query_idx      : the pair index (old object index)
    step           : row['step']  (last object index = step - 1)
    """
    last_idx = int(step) - 1
    old_idx  = int(query_idx)

    # Try recompute first
    if all_aabbs and last_idx in all_aabbs and old_idx in all_aabbs:
        return _determine_situation_recomputed(all_aabbs[last_idx], all_aabbs[old_idx])

    # Fallback: stored label
    if bbox_diff_dict:
        q_info = bbox_diff_dict.get(old_idx) or bbox_diff_dict.get(str(old_idx))
        if q_info and 'situation' in q_info:
            return q_info['situation']

    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# EXISTING HELPERS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def setup_directories(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created output root directory: {path}")

def parse_array_string(string_val):
    try:
        if isinstance(string_val, (list, dict)):
            return string_val
        if pd.isna(string_val):
            return None
        return ast.literal_eval(str(string_val))
    except (ValueError, SyntaxError):
        return None

def load_data(filepath):
    print(f"--> Loading: {filepath}")
    if not os.path.isfile(filepath):
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        return pd.read_excel(filepath)
    encodings = ['utf-8', 'cp1254', 'latin-1']
    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except Exception:
            continue
    print("CRITICAL ERROR: Could not read file.")
    sys.exit(1)

def to_bit_col(val):
    try:    return '1' if float(val) >= 0.5 else '0'
    except: return '0'

def determine_folder_name_new(symbol_list, predicted_list, situations_for_this_group,
                               collapses_for_this_group, actual_collapse_flags):
    base_name = disc(symbol_list)
    last_bit  = "Collapse" if to_bit_col(predicted_list[OUT_DIM]) == '1' else "Stable"

    if actual_collapse_flags:
        act_count      = sum(1 for x in actual_collapse_flags if float(x) >= 0.5)
        act_percentage = int((act_count / len(actual_collapse_flags)) * 100)
    else:
        act_percentage = 0

    if not situations_for_this_group:
        return f"{base_name}_v_no_situation"

    counts = Counter(situations_for_this_group)
    top_3  = counts.most_common(3)
    sit_strings = []
    for sit, count in top_3:
        percentage = int((count / len(situations_for_this_group)) * 100)
        sit_strings.append(f"{sit}_{percentage}")
    dom_sit_combined = "_".join(sit_strings)

    return (f"{base_name}_v_{dom_sit_combined}_{act_percentage}"
            f"_predicted_{last_bit}_{round(predicted_list[OUT_DIM], 2)}")

def get_concat_h(im_list):
    valid_images = [img for img in im_list if img is not None]
    if not valid_images: return None
    dst = valid_images[0]
    for i in range(1, len(valid_images)):
        im        = valid_images[i]
        new_width = dst.width + im.width
        new_height = max(dst.height, im.height)
        new_im    = Image.new('RGB', (new_width, new_height))
        new_im.paste(dst, (0, 0))
        new_im.paste(im, (dst.width, 0))
        dst = new_im
    return dst
def create_text_image(query_idx, length, object_types, situation, size=(256, 256)):
    img = Image.new('RGB', size, color=(40, 44, 52))
    d   = ImageDraw.Draw(img)
    
    try:    q_str = str(int(float(query_idx)))
    except: q_str = str(query_idx)
    try:    l_str = str(int(float(length)))
    except: l_str = str(length)
    
    # Added Situation to the lines list
    lines = [
        f"Query Idx: {q_str}", 
        f"Length: {l_str}", 
        f"Situation: {situation}",
        "", 
        "Object Types:"
    ]
    
    y_offset = 20
    for line in lines:
        # Highlighting the Situation line in a different color (Light Blue)
        fill_color = (135, 206, 250) if "Situation" in line else (255, 255, 255)
        d.text((15, y_offset), line, fill=fill_color)
        y_offset += 25
        
    o_str = str(object_types)
    for i in range(0, len(o_str), 30):
        d.text((15, y_offset), o_str[i:i+30], fill=(255, 204, 0))
        y_offset += 20
    return img

def disc(list):
    list        = ast.literal_eval(str(list))
    list_updated = []
    for sym in list:
        list_updated.append(1 if float(sym) >= 0.5 else 0)
    return str(list_updated)

def disc_collapse(list):
    list = ast.literal_eval(str(list))
    return str(1 if float(list[-1]) >= 0.5 else 0)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def process_images():
    # 1. Load Data
    df_results = load_data(results_file_path)
    df_dataset = load_data(dataset_file_path)

    # 2. Normalize Paths to Absolute
    def to_abs(path):
        if pd.isna(path) or str(path).strip() == "":
            return None
        return os.path.abspath(os.path.join(base_image_path, str(path).lstrip('/')))

    print("Normalizing paths...")
    df_dataset['rgb_image_path'] = df_dataset['rgb_image_path'].apply(to_abs)
    df_dataset['upper_image_path'] = df_dataset['upper_image_path'].apply(to_abs)
    df_results['rgb_image_path'] = df_results['rgb_image_path'].apply(to_abs)

    print("Building dataset indexes...")
    df_dataset_unique = df_dataset.drop_duplicates(subset=['rgb_image_path'], keep='first')
    path_lookup = df_dataset_unique.set_index('rgb_image_path')[
        ['id', 'step', 'object_type', 'upper_image_path', 'bounding_box_differences', 'bbox', 'spatial_relations']
    ].to_dict('index')

    df_dataset['id_step_key'] = list(zip(df_dataset['id'], df_dataset['step']))
    id_step_lookup = df_dataset.set_index('id_step_key').to_dict('index')

    temp_data_storage       = []
    symbol_to_situation_map = {}
    symbol_to_collapse_map  = {}
    symbol_to_actual_collapse_map = {}

    # 3. Analyze Situations
    print(f"\nAnalyzing Situations...")
    for index, row in df_results.iterrows():
        current_rgb_path = row.get('rgb_image_path')
        query_idx        = row.get('query_index')
        symbol_val       = disc(row.get('symbol'))

        actual_val       = row.get('actual')
        actual_list      = parse_array_string(actual_val)
        dataset_row      = path_lookup.get(current_rgb_path)

        if not dataset_row:
            continue

        bbox_diff_raw  = dataset_row.get('bounding_box_differences')
        bbox_col_raw   = dataset_row.get('bbox')
        bbox_diff_dict = parse_array_string(bbox_diff_raw)
        all_aabbs      = parse_array_string(bbox_col_raw) or {}

        true_c_scan = 1 if actual_list and float(actual_list[OUT_DIM]) >= 0.5 else 0
        situation   = "collapse" if true_c_scan == 1 else _get_situation(bbox_diff_dict, all_aabbs, query_idx, dataset_row['step'])
        collapse  = disc_collapse(row.get('predicted'))

        if symbol_val not in symbol_to_situation_map:
            symbol_to_situation_map[symbol_val] = []
        symbol_to_situation_map[symbol_val].append(situation)

        if symbol_val not in symbol_to_collapse_map:
            symbol_to_collapse_map[symbol_val] = []
        symbol_to_collapse_map[symbol_val].append(collapse)

        if symbol_val not in symbol_to_actual_collapse_map:
            symbol_to_actual_collapse_map[symbol_val] = []
        symbol_to_actual_collapse_map[symbol_val].append(actual_list[OUT_DIM])

        temp_data_storage.append({
            'index': index, 'row': row, 'dataset_row': dataset_row,
            'situation': situation, 'all_aabbs': all_aabbs, 'bbox_diff_dict': bbox_diff_dict,
        })

    num_unique_symbols = len(symbol_to_situation_map)
    symbol_stats       = {}
    total_valid_rows   = 0
    pos_weight_tensor  = torch.tensor(POS_WEIGHT_COL)

    # 4. Processing and Saving
    print(f"\nProcessing Images...")
    for item in temp_data_storage:
        try:
            row, dataset_row = item['row'], item['dataset_row']
            query_idx, current_rgb_path = row.get('query_index'), row.get('rgb_image_path')
            
            symbol_list    = parse_array_string(disc(row.get('symbol')))
            predicted_list = parse_array_string(row.get('predicted'))
            actual_list    = parse_array_string(row.get('actual'))

            subfolder_name = determine_folder_name_new(
                symbol_list, predicted_list,
                symbol_to_situation_map[disc(row.get('symbol'))],
                symbol_to_collapse_map[disc(row.get('symbol'))],
                symbol_to_actual_collapse_map[disc(row.get('symbol'))]
            )

            # Recomputed situation for stat & label
            # With:
            actual_list        = parse_array_string(row.get('actual'))
            true_c             = 1 if actual_list and float(actual_list[OUT_DIM]) >= 0.5 else 0
            situation_for_stat = "collapse" if true_c == 1 else _get_situation(item['bbox_diff_dict'], item['all_aabbs'], query_idx, dataset_row['step'])
                        
            # Metadata Filename logic
            base_filename, ext = os.path.splitext(os.path.basename(current_rgb_path))
            unique_filename = f"{base_filename}_q{query_idx}{ext}"
            dest = os.path.join(output_root_dir, subfolder_name)
            os.makedirs(dest, exist_ok=True)
            save_path = os.path.join(dest, unique_filename)

            tower_id, current_step = dataset_row['id'], dataset_row['step']
            related_row = id_step_lookup.get((tower_id, query_idx + 1))
            prev_row    = id_step_lookup.get((tower_id, current_step - 1))

            if SAVE_STITCHED_IMAGES:
                # [Same logic as before, using Save_path]
                paths = [
                    related_row['upper_image_path'] if related_row else None,
                    dataset_row['upper_image_path'],
                    prev_row['rgb_image_path'] if prev_row else None,
                    current_rgb_path
                ]
                imgs = [Image.open(p) if p and os.path.exists(p) else Image.new('RGB', (256, 256), (0,0,0)) for p in paths]
                final_img = get_concat_h(imgs)
                if final_img: final_img.save(save_path)
            else:
                obj_list = []
                if related_row: obj_list.append(str(related_row.get('object_type')))
                if dataset_row: obj_list.append(str(dataset_row.get('object_type')))
                combined_obj_str = " + ".join(obj_list) if obj_list else "None"

                # UPDATED: Added situation_for_stat here
                metadata_img = create_text_image(query_idx, row.get('length'), combined_obj_str, situation_for_stat)
                
                imgs = [metadata_img]
                p_path = prev_row['rgb_image_path'] if prev_row else None
                imgs.append(Image.open(p_path) if p_path and os.path.exists(p_path) else Image.new('RGB', (256, 256), (30,30,30)))
                imgs.append(Image.open(current_rgb_path) if current_rgb_path and os.path.exists(current_rgb_path) else Image.new('RGB', (256, 256), (0,0,0)))

                final_img = get_concat_h(imgs)
                if final_img: final_img.save(save_path)

            total_valid_rows += 1
        except Exception as e:
            print(f"Error at index {item['index']}: {e}")

    print(f"\nDone! Organised images are in: {output_root_dir}")


def analyze_fp_fn_cases(df_results, df_dataset, save_csv=True):
    print("\n" + "=" * 90)
    print("FP / FN ANALYSIS BY SITUATION, SYMBOL, LENGTH, OBJECT TYPE")
    print("=" * 90)

    def to_abs(path):
        if pd.isna(path) or str(path).strip() == "":
            return None
        return os.path.abspath(os.path.join(base_image_path, str(path).lstrip('/')))

    df_dataset = df_dataset.copy()
    df_results = df_results.copy()

    df_dataset["rgb_image_path"] = df_dataset["rgb_image_path"].apply(to_abs)
    df_results["rgb_image_path"]  = df_results["rgb_image_path"].apply(to_abs)

    # ── need upper_image_path + id_step for image stitching ──────────────────
    df_dataset["upper_image_path"] = df_dataset["upper_image_path"].apply(to_abs)
    df_dataset["id_step_key"]      = list(zip(df_dataset["id"], df_dataset["step"]))
    id_step_lookup_local           = df_dataset.set_index("id_step_key").to_dict("index")

    df_dataset_unique = df_dataset.drop_duplicates(subset=["rgb_image_path"], keep="first")
    path_lookup = df_dataset_unique.set_index("rgb_image_path")[
        ["id", "step", "object_type", "bounding_box_differences", "bbox", "upper_image_path"]
    ].to_dict("index")

    records = []

    for idx, row in df_results.iterrows():
        predicted_list = parse_array_string(row.get("predicted"))
        actual_list    = parse_array_string(row.get("actual"))

        if predicted_list is None or actual_list is None:
            continue

        pred_c = 1 if float(predicted_list[OUT_DIM]) >= 0.5 else 0
        true_c = 1 if float(actual_list[OUT_DIM])    >= 0.5 else 0

        if   pred_c == 1 and true_c == 1: err_type = "TP"
        elif pred_c == 0 and true_c == 0: err_type = "TN"
        elif pred_c == 1 and true_c == 0: err_type = "FP"
        else:                             err_type = "FN"

        current_rgb_path = row.get("rgb_image_path")
        dataset_row      = path_lookup.get(current_rgb_path)

        situation   = "unknown"
        object_type = "unknown"
        step        = row.get("length")
        query_idx   = row.get("query_index")

        if dataset_row:
            bbox_diff_dict = parse_array_string(dataset_row.get("bounding_box_differences"))
            all_aabbs      = parse_array_string(dataset_row.get("bbox")) or {}
            step           = dataset_row.get("step")
            object_type    = dataset_row.get("object_type", "unknown")
            situation      = "collapse" if true_c == 1 else _get_situation(bbox_diff_dict, all_aabbs, query_idx, step)

        symbol = disc(row.get("symbol")) if "symbol" in row else "no_symbol"

        records.append({
            "index": idx, "row": row, "dataset_row": dataset_row,
            "error_type": err_type,
            "pred_collapse": pred_c, "true_collapse": true_c,
            "pred_prob": float(predicted_list[OUT_DIM]),
            "true_value": float(actual_list[OUT_DIM]),
            "symbol": symbol, "situation": situation,
            "query_index": query_idx, "length": step,
            "object_type": object_type,
            "rgb_image_path": current_rgb_path,
            "predicted": predicted_list, "actual": actual_list,
        })

    df_err = pd.DataFrame([{k: v for k, v in r.items() if k not in ("row", "dataset_row")} for r in records])

    if df_err.empty:
        print("No valid rows found.")
        return df_err

    # ── Console summaries ─────────────────────────────────────────────────────
    print("\nOverall confusion counts:")
    print(df_err["error_type"].value_counts())

    print("\nFP/FN by situation:")
    print(df_err[df_err["error_type"].isin(["FP","FN"])].groupby(["error_type","situation"]).size().sort_values(ascending=False))

    print("\nFP/FN by symbol:")
    print(df_err[df_err["error_type"].isin(["FP","FN"])].groupby(["error_type","symbol"]).size().sort_values(ascending=False))

    print("\nFP/FN by length:")
    print(df_err[df_err["error_type"].isin(["FP","FN"])].groupby(["error_type","length"]).size().sort_values(ascending=False))

    print("\nFP/FN by object type:")
    print(df_err[df_err["error_type"].isin(["FP","FN"])].groupby(["error_type","object_type"]).size().sort_values(ascending=False))

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    if save_csv:
        out_csv = os.path.join(output_root_dir, "fp_fn_analysis.csv")
        df_err.to_csv(out_csv, index=False)
        print(f"\nSaved full FP/FN analysis to: {out_csv}")

        df_err[df_err["error_type"] == "FP"].to_csv(os.path.join(output_root_dir, "false_positives.csv"), index=False)
        df_err[df_err["error_type"] == "FN"].to_csv(os.path.join(output_root_dir, "false_negatives.csv"), index=False)
        print(f"Saved FP/FN CSVs.")

    # ── Save FP images (predicted collapse but actually stable) ───────────────
    fp_image_root = os.path.join(output_root_dir, "fp_images")
    os.makedirs(fp_image_root, exist_ok=True)
    fp_records    = [r for r in records if r["error_type"] == "FP"]
    print(f"\nSaving FP images ({len(fp_records)} cases) to: {fp_image_root}")

    for r in fp_records:
        try:
            row         = r["row"]
            dataset_row = r["dataset_row"]
            query_idx   = r["query_index"]
            situation   = r["situation"]           # geometric label (never "collapse" for FP)
            current_rgb_path = r["rgb_image_path"]

            if dataset_row is None or current_rgb_path is None:
                continue

            tower_id     = dataset_row["id"]
            current_step = dataset_row["step"]

            # subfolder = geometric situation so FPs are grouped by WHY they fooled the model
            subfolder = os.path.join(fp_image_root)
            os.makedirs(subfolder, exist_ok=True)

            base_filename, ext = os.path.splitext(os.path.basename(current_rgb_path))
            save_path = os.path.join(subfolder, f"{base_filename}_q{query_idx}{ext}")

            prev_row = id_step_lookup_local.get((tower_id, current_step - 1))

            # Build same 3-panel image: [metadata | prev frame | current frame]
            obj_list = []
            related_row = id_step_lookup_local.get((tower_id, query_idx + 1))
            if related_row: obj_list.append(str(related_row.get("object_type")))
            obj_list.append(str(dataset_row.get("object_type")))
            combined_obj_str = " + ".join(obj_list) if obj_list else "None"

            metadata_img = create_text_image(query_idx, row.get("length"), combined_obj_str, situation)

            p_path = prev_row["rgb_image_path"] if prev_row else None
            imgs = [
                metadata_img,
                Image.open(p_path) if p_path and os.path.exists(p_path) else Image.new("RGB", (256, 256), (30, 30, 30)),
                Image.open(current_rgb_path) if current_rgb_path and os.path.exists(current_rgb_path) else Image.new("RGB", (256, 256), (0, 0, 0)),
            ]

            final_img = get_concat_h(imgs)
            if final_img:
                final_img.save(save_path)

        except Exception as e:
            print(f"  Error saving FP image at index {r['index']}: {e}")

    print(f"FP images saved. Structure: fp_images/<situation>/")
    print("=" * 90)
    return df_err

def print_xlsx_collapse_metrics(df_results):
    tp = tn = fp = fn = 0

    for _, row in df_results.iterrows():
        predicted_list = parse_array_string(row.get("predicted"))
        actual_list    = parse_array_string(row.get("actual"))

        if predicted_list is None or actual_list is None:
            continue

        pred_c = 1 if float(predicted_list[OUT_DIM]) >= 0.5 else 0
        true_c = 1 if float(actual_list[OUT_DIM]) >= 0.5 else 0

        if pred_c == 1 and true_c == 1:
            tp += 1
        elif pred_c == 0 and true_c == 0:
            tn += 1
        elif pred_c == 1 and true_c == 0:
            fp += 1
        elif pred_c == 0 and true_c == 1:
            fn += 1

    eps = 1e-12
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    tpr = tp / (tp + fn + eps)
    tnr = tn / (tn + fp + eps)
    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)

    print("\n" + "=" * 70)
    print("COLLAPSE METRICS FROM XLSX PREDICTED/ACTUAL VALUES")
    print("=" * 70)
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print("-" * 70)
    print(f"ACC:       {acc:.4f}")
    print(f"TPR/Recall:{tpr:.4f}")
    print(f"TNR:       {tnr:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"FNR:       {fnr:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1:        {f1:.4f}")
    print("=" * 70)


def analyze_object_symbols(df_results, df_dataset, save_csv=True):
    print("\n" + "=" * 90)
    print("OBJECT SYMBOL ANALYSIS")
    print("For each obj_symbol_query: percentage of real object_type + object_size_or_path")
    print("=" * 90)

    def to_abs(path):
        if pd.isna(path) or str(path).strip() == "":
            return None
        return os.path.abspath(os.path.join(base_image_path, str(path).lstrip('/')))

    def get_obj_symbol_col(df):
        candidates = [
            "obj_symbol_query",
            "obj_Symbol_query",
            "object_symbol_query",
            "object_Symbol_query",
        ]
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def disc_obj_symbol(val):
        """
        Converts object symbol into stable discrete string.
        Works whether obj_symbol_query is:
        - a list-like string: "[0.2, 0.9]"
        - an actual list
        - a scalar symbol id
        """
        parsed = parse_array_string(val)

        if isinstance(parsed, (list, tuple, np.ndarray)):
            return str([1 if float(x) >= 0.5 else 0 for x in parsed])

        try:
            # scalar case
            return str(int(float(val)))
        except Exception:
            return str(val)

    def object_name_from_row(r):
        obj_type = str(r.get("object_type", "unknown"))

        if "object_size_or_path" in r:
            obj_size = r.get("object_size_or_path", "unknown")
        elif "object_size" in r:
            obj_size = r.get("object_size", "unknown")
        else:
            obj_size = "unknown"

        return f"{obj_type} | {obj_size}"

    df_results = df_results.copy()
    df_dataset = df_dataset.copy()

    obj_symbol_col = get_obj_symbol_col(df_results)

    if obj_symbol_col is None:
        print("Could not find obj_symbol_query column in df_results.")
        print("Available result columns:")
        print(list(df_results.columns))
        return pd.DataFrame()

    print(f"Using object symbol column: {obj_symbol_col}")

    df_dataset["rgb_image_path"] = df_dataset["rgb_image_path"].apply(to_abs)
    df_results["rgb_image_path"] = df_results["rgb_image_path"].apply(to_abs)

    df_dataset["id_step_key"] = list(zip(df_dataset["id"], df_dataset["step"]))

    needed_cols = ["id", "step", "object_type"]

    if "object_size_or_path" in df_dataset.columns:
        needed_cols.append("object_size_or_path")
    elif "object_size" in df_dataset.columns:
        needed_cols.append("object_size")
    else:
        print("WARNING: No object_size_or_path or object_size column found in dataset.")
        print("Only object_type will be used.")

    df_dataset_unique = df_dataset.drop_duplicates(subset=["rgb_image_path"], keep="first")
    path_lookup = df_dataset_unique.set_index("rgb_image_path")[["id", "step"]].to_dict("index")

    id_step_lookup = df_dataset.set_index("id_step_key")[needed_cols].to_dict("index")

    records = []

    for idx, row in df_results.iterrows():
        current_rgb_path = row.get("rgb_image_path")
        query_idx = row.get("query_index")

        if pd.isna(query_idx):
            continue

        dataset_row = path_lookup.get(current_rgb_path)

        if dataset_row is None:
            continue

        try:
            query_step = int(float(query_idx)) + 1
        except Exception:
            continue

        tower_id = dataset_row["id"]
        query_object_row = id_step_lookup.get((tower_id, query_step))

        if query_object_row is None:
            continue

        obj_symbol = disc_obj_symbol(row.get(obj_symbol_col))
        real_object = object_name_from_row(query_object_row)

        records.append({
            "result_index": idx,
            "tower_id": tower_id,
            "query_index": query_idx,
            "query_step": query_step,
            "obj_symbol_query": obj_symbol,
            "real_object": real_object,
            "object_type": query_object_row.get("object_type", "unknown"),
            "object_size_or_path": query_object_row.get(
                "object_size_or_path",
                query_object_row.get("object_size", "unknown")
            ),
        })

    df_obj = pd.DataFrame(records)

    if df_obj.empty:
        print("No valid rows found for object symbol analysis.")
        return df_obj

    counts = (
        df_obj
        .groupby(["obj_symbol_query", "real_object"])
        .size()
        .reset_index(name="count")
    )

    totals = (
        df_obj
        .groupby("obj_symbol_query")
        .size()
        .reset_index(name="total_for_symbol")
    )

    summary = counts.merge(totals, on="obj_symbol_query")
    summary["percentage"] = 100.0 * summary["count"] / summary["total_for_symbol"]

    summary = summary.sort_values(
        ["obj_symbol_query", "percentage", "count"],
        ascending=[True, False, False]
    )

    print("\nObject distribution per obj_symbol_query:")
    for sym, group in summary.groupby("obj_symbol_query"):
        total = int(group["total_for_symbol"].iloc[0])
        print("\n" + "-" * 90)
        print(f"Object symbol: {sym}   total rows: {total}")
        print("-" * 90)

        for _, r in group.iterrows():
            print(
                f"{r['real_object']:<55} "
                f"{int(r['count']):>5} / {int(r['total_for_symbol']):<5} "
                f"= {r['percentage']:6.2f}%"
            )

    if save_csv:
        out_raw = os.path.join(output_root_dir, "object_symbol_raw_rows.csv")
        out_summary = os.path.join(output_root_dir, "object_symbol_distribution.csv")

        df_obj.to_csv(out_raw, index=False)
        summary.to_csv(out_summary, index=False)

        print(f"\nSaved raw object-symbol rows to: {out_raw}")
        print(f"Saved object-symbol distribution to: {out_summary}")

    print("=" * 90)

    return summary

if __name__ == "__main__":
    setup_directories(output_root_dir)

    df_results = load_data(results_file_path)
    df_dataset = load_data(dataset_file_path)

    print_xlsx_collapse_metrics(df_results)

    df_fp_fn = analyze_fp_fn_cases(df_results, df_dataset, save_csv=True)
    df_obj_symbol = analyze_object_symbols(df_results, df_dataset, save_csv=True)

    process_images()