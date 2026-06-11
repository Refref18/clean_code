"""
plan_for_shortest_tallest.py
============================
For every object group where  min_levels == target_min_level
                          AND  max_levels == target_max_level
(computed in-memory — no pre-generated CSV needed):

  Run the PDDL planner TWICE per group:
    • Run A  →  goal height = target_min_level  (shortest)
    • Run B  →  goal height = target_max_level  (tallest)

  After each run:
    1. Extract the planner's object ordering from the plan.
    2. Check whether it matches a known real sequence at that height.
    3. Run the ordered objects through the physics simulator.
    4. Compute simulated height using stepwise-level logic and compare.

  Results saved to  save/<model_name>/eval/results.csv

Usage:
    python plan_for_shortest_tallest.py -c configs/eval.yaml
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import load_ckpt
from symbol_semantics import get_semantic_symbols
from data_collection_direct.exp import get_experiment
import load_data as _load_data
from load_data import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# CLI + config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    name = cfg["model_name"]
    save_root         = os.path.join("save", name)
    cfg["_domain"]    = os.path.join(save_root, "domain.pddl")
    save_root    = os.path.join("save", f"{name}")
    eval_dir          = os.path.join(save_root, "eval")
    cfg["_pddl_dir"]  = os.path.join(save_root, "PDDL_FILES")
    cfg["_plan_file"] = os.path.join(save_root, "PDDL_FILES", "sas_plan")
    cfg["_eval_dir"]  = eval_dir
    cfg["_exp_dir"]   = os.path.join(eval_dir, "sim_experiments")
    cfg["_img_dir"]   = os.path.join(eval_dir, "images")
    cfg["_results"]   = os.path.join(eval_dir, "results.csv")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(filepath):
    for enc in ['utf-8', 'cp1254', 'latin-1']:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except Exception:
            continue
    sys.exit(f"CRITICAL: Cannot read {filepath}")


def obj_to_canonical(obj_type, obj_size_or_path):
    try:
        float(obj_size_or_path)
        return (str(obj_type), str(obj_size_or_path))
    except (ValueError, TypeError):
        clean = str(obj_size_or_path).split('/')[-1].replace('.urdf', '')
        return (str(obj_type), clean)


# ─────────────────────────────────────────────────────────────────────────────
# Object symbol computation  (replaces problem.py's get_obj_symbols)
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_SIZES = [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13,
               0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]


def _clean_size(obj_size_or_path):
    """Normalise a raw size/path string to a clean filename stem."""
    try:
        val = float(obj_size_or_path)
        return str(min(KNOWN_SIZES, key=lambda x: abs(x - val)))
    except (ValueError, TypeError):
        return str(obj_size_or_path).split('/')[-1].replace('.urdf', '')


def load_depth_image(obj_type, obj_size_or_path, depth_dir, device):
    """
    Load upper + lower depth images for one object.
    Returns a [2, 64, 64] tensor, or zeros if files are missing.
    """
    clean    = _clean_size(obj_size_or_path)
    base     = f"{obj_type}_{clean}"
    upper_p  = os.path.join(depth_dir, f"upper_{base}_new.npy")
    lower_p  = os.path.join(depth_dir, f"lower_{base}_new.npy")

    if os.path.exists(upper_p) and os.path.exists(lower_p):
        u = torch.from_numpy(np.load(upper_p)).unsqueeze(0).float()
        l = torch.from_numpy(np.load(lower_p)).unsqueeze(0).float()
        return torch.cat([u, l], dim=0).to(device)
    else:
        print(f"  [WARN] Depth images missing for {base} — using zeros")
        return torch.zeros(2, 64, 64, device=device)


def get_obj_symbols(ordered_objects, model, cfg):
    """
    Compute binary object symbols for a list of (type, path) tuples.
    Uses model.get_object_symbols() directly — no problem.py needed.

    Returns list of lists, one per object: [[0,1,1,0], ...]
    """
    depth_dir = cfg["data"]["depth_image_dir"]
    device    = next(model.parameters()).device

    imgs = [load_depth_image(otype, opath, depth_dir, device)
            for otype, opath in ordered_objects]

    # Stack into a single batch [N, 2, 64, 64] and run encoder
    x = torch.stack(imgs, dim=0)   # [N, 2, 64, 64]
    model.eval()
    model.gs_obj_layer.hard          = True
    model.gs_obj_layer.deterministic = True
    with torch.no_grad():
        bits, _ = model.get_object_symbols(x)   # [N, obj_sym_size]

    return bits.cpu().int().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Collapse symbol computation  (replaces problem.py's get_relation_symbols)
# ─────────────────────────────────────────────────────────────────────────────

def get_collapse_symbols(model, cfg):
    train_cfg = cfg["_train_cfg"]
    rel_dim   = train_cfg["model"]["symbol_size"]
    out_dim   = 4
    device    = next(model.parameters()).device

    collapse_syms, inserted_syms, normal_sym = get_semantic_symbols(
        model,
        sym_size=rel_dim,
        out_dim=out_dim,
        collapse_threshold=0.5,
        device=device,
    )
    return [list(s) for s in collapse_syms]

# ─────────────────────────────────────────────────────────────────────────────
# Height analysis  (inlined from height_analysis.py — runs in memory)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_col(val):
    try:
        if isinstance(val, (list, dict)):
            return val
        if pd.isna(val):
            return None
        return ast.literal_eval(str(val))
    except Exception:
        return None


def _extract_object_name(path):
    if pd.isna(path):
        return "unknown"
    filename = os.path.basename(str(path)).replace('.urdf', '')
    parts    = filename.split('_')
    return "_".join(parts[:3]) if len(parts) >= 3 else parts[0]


def _object_key(row):
    clean = _extract_object_name(row.get('object_size_or_path'))
    return str(row.get('object_type', 'unknown')) + "_" + clean


def compute_levels_stepwise(step_bboxes, threshold):
    sorted_steps = sorted(step_bboxes.keys())
    if not sorted_steps:
        return 0
    levels = 0
    for s in sorted_steps:
        bbox = step_bboxes[s]
        if not bbox:
            continue
        new_idx = s - 1
        if new_idx not in bbox:
            new_idx = max(bbox.keys())
        new_zmax = bbox[new_idx]['max'][2]
        if levels == 0:
            levels = 1
        else:
            prev_zmaxes = [bbox[i]['max'][2] for i in bbox if i != new_idx]
            if prev_zmaxes and new_zmax > max(prev_zmaxes) + threshold:
                levels += 1
    return levels


def _compute_physical_height(step_bboxes):
    if not step_bboxes:
        return 0.0
    final_bbox = step_bboxes[max(step_bboxes.keys())]
    if not final_bbox:
        return 0.0
    return round(
        max(v['max'][2] for v in final_bbox.values()) -
        min(v['min'][2] for v in final_bbox.values()), 4)


def compute_height_analysis(dataset_csv, z_threshold):
    """
    Runs height analysis in memory.
    Returns (df_details, df_summary) — no CSV files written.
    """
    df = load_csv(dataset_csv)
    print(f"  [height_analysis] Loaded {len(df)} rows")

    tower_objects   = defaultdict(dict)
    tower_bboxes    = defaultdict(dict)
    tower_collapsed = {}

    for _, row in df.iterrows():
        tid  = row['id']
        step = int(row['step'])
        tower_objects[tid][step] = _object_key(row)
        bbox = _parse_col(row.get('bbox'))
        if bbox:
            tower_bboxes[tid][step] = bbox
        collapse_val = row.get('collapse', 0)
        try:
            if float(collapse_val) >= 0.5:
                tower_collapsed[tid] = True
        except (TypeError, ValueError):
            pass

    records = []
    for tid, step_map in tower_objects.items():
        ordered_objs = [step_map[s] for s in sorted(step_map.keys())]
        object_group = tuple(sorted(ordered_objs))
        collapsed    = tower_collapsed.get(tid, False)
        step_bboxes  = tower_bboxes.get(tid, {})
        levels  = compute_levels_stepwise(step_bboxes, z_threshold) if not collapsed else None
        phys_h  = _compute_physical_height(step_bboxes)             if not collapsed else None
        records.append({
            'tower_id':        tid,
            'n_objects':       len(ordered_objs),
            'object_group':    str(object_group),
            'object_order':    str(tuple(ordered_objs)),
            'collapsed':       collapsed,
            'levels':          levels,
            'physical_height': phys_h,
        })

    df_details = pd.DataFrame(records)
    df_valid   = df_details[df_details['collapsed'] == False].copy()

    if df_valid.empty:
        print("  [height_analysis] WARNING: no non-collapsed towers found.")
        return df_details, pd.DataFrame()

    agg = df_valid.groupby('object_group').agg(
        n_towers            =('tower_id',        'count'),
        n_objects           =('n_objects',        'first'),
        min_levels          =('levels',           'min'),
        max_levels          =('levels',           'max'),
        mean_levels         =('levels',           'mean'),
        min_physical_height =('physical_height',  'min'),
        max_physical_height =('physical_height',  'max'),
    ).reset_index()

    df_cc = (df_details[df_details['collapsed'] == True]
             .groupby('object_group').size()
             .rename('n_collapsed').reset_index())
    agg = agg.merge(df_cc, on='object_group', how='left')
    agg['n_collapsed'] = agg['n_collapsed'].fillna(0).astype(int)

    idx_min  = df_valid.groupby('object_group')['levels'].idxmin()
    idx_max  = df_valid.groupby('object_group')['levels'].idxmax()
    min_rows = df_valid.loc[idx_min, ['object_group', 'object_order', 'tower_id']].rename(
        columns={'object_order': 'min_level_order', 'tower_id': 'min_level_tower_id'})
    max_rows = df_valid.loc[idx_max, ['object_group', 'object_order', 'tower_id']].rename(
        columns={'object_order': 'max_level_order', 'tower_id': 'max_level_tower_id'})

    df_summary = (agg
                  .merge(min_rows, on='object_group', how='left')
                  .merge(max_rows, on='object_group', how='left'))
    df_summary['mean_levels'] = df_summary['mean_levels'].round(2)
    print(f"  [height_analysis] {len(df_details)} towers | {len(df_summary)} groups")
    return df_details, df_summary


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_pair_to_original(df_dataset):
    """object_pair_string → (object_type, object_size_or_path)"""
    lookup = {}
    for _, row in df_dataset.iterrows():
        otype    = str(row.get('object_type', ''))
        raw_path = str(row.get('object_size_or_path', ''))
        filename = os.path.basename(raw_path).replace('.urdf', '')
        parts    = filename.split('_')
        clean    = "_".join(parts[:3]) if len(parts) >= 3 else parts[0]
        key      = f"{otype}_{clean}"
        if key not in lookup:
            lookup[key] = (otype, row.get('object_size_or_path'))
    return lookup


def group_str_to_object_list(group_str, pair_to_original):
    try:
        pair_keys = ast.literal_eval(group_str)
    except Exception as e:
        print(f"  [WARN] Cannot parse group: {group_str} — {e}")
        return None, None
    obj_list   = []
    pair_order = []
    for key in pair_keys:
        original = pair_to_original.get(key)
        if original is None:
            print(f"  [WARN] No original for key: {key}")
            return None, None
        obj_list.append(original)
        pair_order.append(key)
    return obj_list, pair_order


def build_known_sequences(df_details, group_label, target_height):
    subset = df_details[
        (df_details['object_group'] == group_label) &
        (df_details['levels']       == target_height) &
        (df_details['collapsed']    == False)
    ]
    known = set()
    for _, row in subset.iterrows():
        try:
            known.add(tuple(ast.literal_eval(str(row['object_order']))))
        except Exception:
            pass
    return known


# ─────────────────────────────────────────────────────────────────────────────
# PDDL problem.pddl construction  (replaces problem.py's construct_domain)
# ─────────────────────────────────────────────────────────────────────────────

def _object_sym_to_pddl(obj_symbols, obj_dim, indentation="\t\t"):
    schema = ""
    for obj_num, obj_val in enumerate(obj_symbols):
        name    = f"obj{obj_num}"
        schema += f"{indentation}(top-0 {name})\n"
        schema += indentation
        for j, val in enumerate(obj_val):
            schema += f"(z{j} {name}) " if val == 1 else f"(not_z{j} {name}) "
        schema += f"(not_z{obj_dim} {name}) \n"
    return schema


def _initial_relation_sym_to_pddl(n_objs, rel_dim, indentation="\t\t"):
    schema = ""
    for i in range(n_objs):
        for j in range(n_objs):
            if i == j:
                continue
            schema += indentation
            for k in range(rel_dim + 1):
                schema += f"(not_r{k} obj{i} obj{j}) "
            schema += "\n"
    return schema


def _spec_pairs(all_obj_syms, spec_obj_syms):
    all_possible_indices = []
    for target in spec_obj_syms:
        indices = [i for i, val in enumerate(all_obj_syms) if val == target]
        all_possible_indices.append(indices)
    return [tuple(f"obj{idx}" for idx in combo)
            for combo in product(*all_possible_indices)]


def _get_spec_possibilities(rel_sym, all_obj_syms, spec_obj_syms,
                             rel_dim, indentation="\t\t\t"):
    schema = indentation + "(or\n"
    for (n1, n2) in _spec_pairs(all_obj_syms, spec_obj_syms):
        schema += indentation + "\t(and "
        for k, val in enumerate(rel_sym):
            schema += f"(r{k} {n1} {n2}) " if val == 1 else f"(not_r{k} {n1} {n2}) "
        schema += f"(r{rel_dim} {n1} {n2}) )\n"
    schema += indentation + ")\n"
    return schema


def _goal_with_obj_restriction(req_sym_list, all_obj_syms, rel_dim, indentation="\t\t"):
    schema = ""
    for case in req_sym_list:
        for possible_sym in case["pos_rel_symbols"]:
            schema += indentation + "(or\n"
            schema += _get_spec_possibilities(
                possible_sym, all_obj_syms, case["obj_symbols"], rel_dim)
            schema += indentation + ")\n"
    return schema


def construct_problem_pddl(obj_symbols, collapse_syms, req_sym_list,
                            rel_dim, obj_dim, target_height,
                            save_dir):
    """
    Build and write problem.pddl.
    Replaces problem.py's construct_domain() + goal_intro() entirely.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_objs = len(obj_symbols)

    domain  = "(define (problem blocks-problem)\n"
    domain += "\t(:domain blocks)\n"
    domain += "\t(:objects\n\t\t"
    domain += " ".join(f"obj{i}" for i in range(n_objs)) + " - object\n\t)\n"
    domain += "\t(:init\n"
    domain += f"\t\t(H0)\n"
    domain += f"\t\t(active-count-0)\n"
    domain += _object_sym_to_pddl(obj_symbols, obj_dim)
    domain += _initial_relation_sym_to_pddl(n_objs, rel_dim)
    domain += "\t)\n"

    # Goal
    domain += "\t(:goal (and\n"
    domain += f"\t\t(H{target_height})\n"
    domain += f"\t\t(all-used)\n"
    domain += f"\t\t(not (active-count-collapse))\n"
    domain += _goal_with_obj_restriction(req_sym_list, obj_symbols, rel_dim)
    domain += "\t))\n)"

    out_path = os.path.join(save_dir, "problem.pddl")
    with open(out_path, "w") as f:
        f.write(domain)
    print(f"  → problem.pddl written  (goal: H{target_height})")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Simulation height reading
# ─────────────────────────────────────────────────────────────────────────────

def _parse_bbox_col(val):
    try:
        if isinstance(val, dict):
            return val
        if pd.isna(val):
            return None
        return ast.literal_eval(str(val))
    except Exception:
        return None


def read_sim_levels(experiment_csv_path, z_threshold):
    actual_path = experiment_csv_path
    if not os.path.exists(actual_path):
        fallback = experiment_csv_path.replace('.csv', 'worker.csv')
        if os.path.exists(fallback):
            actual_path = fallback
        else:
            print(f"  [SIM] Result CSV not found: {experiment_csv_path}")
            return None, True

    try:
        df_sim = load_csv(actual_path)   # ← this line was missing
    except Exception as e:
        print(f"  [SIM] Cannot read CSV: {e}")
        return None, True

    step_bboxes = {}
    collapsed   = False
    for _, row in df_sim.iterrows():
        step = int(row.get('step', 0))
        cv   = row.get('collapse', 0)
        try:
            if float(cv) >= 0.5:
                collapsed = True
        except (TypeError, ValueError):
            if str(cv).strip().lower() in ('true', '1', 'yes'):
                collapsed = True
        bbox = _parse_bbox_col(row.get('bbox'))
        if bbox:
            step_bboxes[step] = {int(k): v for k, v in bbox.items()}

    if collapsed:
        return None, True
    return compute_levels_stepwise(step_bboxes, z_threshold), False
# ─────────────────────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────────────────────

def run_planner(cfg):
    problem_path = os.path.join(cfg["_pddl_dir"], "problem.pddl")
    plan_file    = cfg["_plan_file"]
    if os.path.exists(plan_file):
        os.remove(plan_file)

    cmd = [sys.executable, cfg["planner"]["fast_downward"],
           "--plan-file", plan_file,
           cfg["_domain"], problem_path,
           "--search", cfg["planner"]["search"]]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout + result.stderr

    if result.returncode == 0 and os.path.exists(plan_file):
        with open(plan_file) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith(';')]
        return "SUCCESS", lines, stdout
    if result.returncode == 12 or "NO PLAN FOUND" in stdout:
        return "NO_PLAN", [], stdout
    return f"ERROR_{result.returncode}", [], stdout


def extract_order_from_plan(plan_lines, pair_order):
    placed = []
    for line in plan_lines:
        line   = line.strip().lstrip('(').rstrip(')')
        tokens = line.split()
        if not tokens or not tokens[0].startswith('a_'):
            continue
        obj_tokens = [t for t in tokens[1:] if re.match(r'^obj\d+$', t)]
        if obj_tokens:
            idx = int(obj_tokens[-1].replace('obj', ''))
            if idx < len(pair_order):
                placed.append(pair_order[idx])
    return tuple(placed) if placed else None


def save_pddl_copy(cfg, group_idx, target_height, status):
    folder  = "successful_plans" if status == "SUCCESS" else "no_plan_files"
    dst_dir = os.path.join(cfg["_pddl_dir"], folder)
    os.makedirs(dst_dir, exist_ok=True)
    label   = f"group_{group_idx}_H{target_height}_{status}"

    src_pddl = os.path.join(cfg["_pddl_dir"], "problem.pddl")
    if os.path.exists(src_pddl):
        shutil.copy(src_pddl, os.path.join(dst_dir, f"{label}.pddl"))

    src_plan = cfg["_plan_file"]           # sas_plan
    if os.path.exists(src_plan):
        shutil.copy(src_plan, os.path.join(dst_dir, f"{label}.sas_plan"))

# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────
def run_simulation(plan_order_result, pair_to_original, cfg, group_idx, target_height):
    """
    plan_order_result: tuple of pair_key strings in PLANNED order
    """
    os.makedirs(cfg["_exp_dir"], exist_ok=True)
    run_label = f"group_{group_idx}_H{target_height}"
    z_thr     = cfg["eval"]["z_level_threshold"]

    # Build ordered object list from the PLAN order, not the group order
    ordered_objects = [pair_to_original[k] for k in plan_order_result]

    print(f"  [SIM] Order (from plan):")
    for k, obj in enumerate(ordered_objects):
        print(f"    [{k}] {obj[0]} | {os.path.basename(str(obj[1]))}")

    try:
        exp_dir = cfg["_exp_dir"] + "/"
        get_experiment(ordered_objects, exp_dir, run_label)
    except Exception as e:
        print(f"  [SIM] Exception: {e}")
        return None, False, True, None

    result_csv            = os.path.join(cfg["_exp_dir"], f"{run_label}.csv")
    sim_levels, collapsed = read_sim_levels(result_csv, z_thr)

    if collapsed:
        print(f"  [SIM] Collapsed")
        return None, False, True, result_csv
    if sim_levels is None:
        return None, False, False, result_csv

    sim_success = (sim_levels == target_height)
    print(f"  [SIM] sim={sim_levels}  target={target_height}  "
          f"→  {'✓' if sim_success else '✗'}")
    return sim_levels, sim_success, False, result_csv


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

_IMG_H = 256; _CAP_H = 120; _BG = (15, 15, 15); _FG = (255, 255, 255)


def get_all_step_rgb_paths(experiment_csv_path):
    # get_experiment writes to {IMAGE_DIR}{case}.csv with no slash,
    # so the real file is {csv_path_without_extension}worker.csv
    # Try both the expected path and the actual get_experiment output path
    candidates = [
        experiment_csv_path,
        experiment_csv_path.replace('.csv', 'worker.csv'),
        os.path.join(os.path.dirname(experiment_csv_path),
                     os.path.basename(experiment_csv_path).replace('.csv', '') + 'worker.csv'),
    ]

    df_sim = None
    for path in candidates:
        if os.path.exists(path):
            try:
                df_sim = load_csv(path)
                break
            except Exception:
                continue

    if df_sim is None:
        print(f"  [IMG] No CSV found, tried: {candidates}")
        return []

    paths = []
    for _, row in df_sim.sort_values('step').iterrows():
        rgb = str(row.get('rgb_image_path', ''))
        if rgb and rgb != 'nan' and os.path.exists(rgb):
            paths.append(rgb)

    print(f"  [IMG] Found {len(paths)} step images")
    return paths


def save_result_image(cfg, group_label, group_idx, target_height,
                      plan_order_str, matched_known,
                      sim_levels, sim_success, sim_collapsed,
                      planner_status, rgb_paths, one_shot_swap=None):
    os.makedirs(cfg["_img_dir"], exist_ok=True)
    target_w = int(_IMG_H * 4 / 3)

    step_imgs = []
    for p in rgb_paths:
        try:
            img   = Image.open(p).convert('RGB')
            ratio = _IMG_H / img.height
            img   = img.resize((max(1, int(img.width * ratio)), _IMG_H), Image.LANCZOS)
            step_imgs.append(img)
        except Exception:
            pass
    if not step_imgs:
        step_imgs.append(Image.new('RGB', (target_w, _IMG_H), (40, 40, 40)))

    cell_w  = max(max(img.width for img in step_imgs), target_w)
    canvas  = Image.new('RGB', (cell_w * len(step_imgs), _IMG_H + _CAP_H), _BG)
    draw    = ImageDraw.Draw(canvas)
    for idx, img in enumerate(step_imgs):
        canvas.paste(img, (idx * cell_w + (cell_w - img.width) // 2, 0))
        draw.text((idx * cell_w + 5, 5), f"step {idx+1}", fill=_FG)

    try:
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        fb = fs = ImageFont.load_default()

    if planner_status != "SUCCESS":
        sc, sl = (255,165,0),  f"NO_PLAN / {planner_status}"
    elif sim_collapsed:
        sc, sl = (180,80,220), "COLLAPSED"
    elif sim_success:
        sc, sl = (80,200,120), f"SUCCESS  (sim={sim_levels} == target={target_height})"
    else:
        sc, sl = (220,80,60),  (f"MISMATCH  (sim={sim_levels} ≠ target={target_height})"
                                 if sim_levels is not None else "SIM ERROR")

    y = _IMG_H + 5
    draw.text((8, y), f"[group {group_idx}  H{target_height}]  {sl}", fill=sc, font=fb); y += 18
    draw.text((8, y), "✓ matches known" if matched_known else "✗ new sequence",
              fill=(160,210,160) if matched_known else (210,160,160), font=fs);        y += 15
    draw.text((8, y), f"order: {plan_order_str[:110]}", fill=(180,180,180), font=fs); y += 15
    draw.text((8, y), group_label[:110], fill=(120,120,120), font=fs)
    if one_shot_swap:
        y += 15
        draw.text((8, y), f"ONE-SHOT: {one_shot_swap[:110]}", fill=(180,130,255), font=fs)

    suffix   = f"_os_{re.sub(r'[^\w]','_',one_shot_swap)}" if one_shot_swap else ""
    out_path = os.path.join(
        cfg["_img_dir"],
        re.sub(r'[^\w\-]', '_',
               f"group_{group_idx}_H{target_height}_{planner_status}{suffix}") + ".png"
    )
    canvas.save(out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# One-shot helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_match_cache(cfg):
    path = cfg["one_shot"]["match_cache_path"]
    if not os.path.exists(path):
        if cfg["one_shot"]["enabled"]:
            sys.exit(f"CRITICAL: one_shot.enabled=true but cache not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def build_one_shot_maps_multi(cfg, match_cache):
    if not match_cache:
        return {}
    one_shot_extra       = [tuple(o) for o in cfg["one_shot"].get("objects", [])]
    matched_to_one_shots = defaultdict(list)
    for obj in one_shot_extra:
        one_shot_canon = obj_to_canonical(*obj)
        result         = match_cache.get(str(one_shot_canon), {})
        if result.get('is_novel', True) or result.get('status') == 'already_exists':
            continue
        try:
            matched_canon = tuple(ast.literal_eval(result.get('best_match', '')))
            matched_to_one_shots[matched_canon].append(obj)
        except Exception:
            pass
    return dict(matched_to_one_shots)


def _make_one_shot_row(group_label, n_objects, target_height, plan_str, matched_known,
                       orig_label, os_label, positions,
                       sim_levels, sim_success, sim_collapsed, image_path):
    return {
        'group': group_label, 'n_objects': n_objects,
        'target_height': target_height, 'status': 'ONE_SHOT',
        'plan_order': plan_str, 'matched_known_sequence': matched_known,
        'sim_levels': sim_levels, 'sim_success': sim_success,
        'sim_collapsed': sim_collapsed, 'image_path': image_path,
        'is_one_shot': True, 'one_shot_object': os_label,
        'swapped_original': orig_label, 'swap_positions': str(positions),
        'planner_output': '',
    }


def run_one_shot_variants(cfg, plan_order_result, pair_to_original,
                           matched_to_one_shots, group_label, n_objects,
                           group_idx, target_height, plan_str, matched_known):
    if not matched_to_one_shots:
        return []

    baseline = [pair_to_original[k] for k in plan_order_result
                if pair_to_original.get(k)]
    if len(baseline) != len(plan_order_result):
        return []

    canon_to_pos = defaultdict(list)
    for pos, orig in enumerate(baseline):
        canon = obj_to_canonical(*orig)
        if canon in matched_to_one_shots:
            canon_to_pos[canon].append(pos)

    if not canon_to_pos:
        return []

    z_thr    = cfg["eval"]["z_level_threshold"]
    os_rows  = []

    for orig_canon, positions in canon_to_pos.items():
        orig_label = f"{orig_canon[0]}_{orig_canon[1]}"
        for one_shot_obj in matched_to_one_shots[orig_canon]:
            os_canon  = obj_to_canonical(*one_shot_obj)
            os_label  = f"{os_canon[0]}_{os_canon[1]}"
            swap_desc = f"{orig_label} → {os_label}  pos={positions}"
            print(f"\n  ── [ONE-SHOT] {swap_desc}  (H{target_height}) ──")

            swapped = list(baseline)
            for pos in positions:
                swapped[pos] = one_shot_obj

            run_label  = (f"group_{group_idx}_H{target_height}"
                          f"_os_{re.sub(r'[^\w]','_',os_label)}")
            os.makedirs(cfg["_exp_dir"], exist_ok=True)
            try:
                get_experiment(swapped, cfg["_exp_dir"] + "/", run_label)
            except Exception as e:
                print(f"  [ONE-SHOT] Sim error: {e}")
                os_rows.append(_make_one_shot_row(
                    group_label, n_objects, target_height, plan_str, matched_known,
                    orig_label, os_label, positions, None, False, True, None))
                continue

            result_csv            = os.path.join(cfg["_exp_dir"], f"{run_label}.csv")
            sim_levels, collapsed = read_sim_levels(result_csv, z_thr)
            sim_success           = (not collapsed and sim_levels == target_height)

            rgb_paths  = get_all_step_rgb_paths(result_csv)
            image_path = save_result_image(
                cfg, group_label, group_idx, target_height,
                plan_str, matched_known, sim_levels, sim_success, collapsed,
                "SUCCESS", rgb_paths, one_shot_swap=swap_desc)

            os_rows.append(_make_one_shot_row(
                group_label, n_objects, target_height, plan_str, matched_known,
                orig_label, os_label, positions,
                sim_levels, sim_success, collapsed, image_path))

    return os_rows


# ─────────────────────────────────────────────────────────────────────────────
# Empty result row helper
# ─────────────────────────────────────────────────────────────────────────────

def _empty_row(group_label, n_objects, target_height, status, planner_output=''):
    return {
        'group': group_label, 'n_objects': n_objects,
        'target_height': target_height, 'status': status,
        'plan_order': '', 'matched_known_sequence': False,
        'sim_levels': None, 'sim_success': False, 'sim_collapsed': None,
        'image_path': None, 'is_one_shot': False,
        'one_shot_object': None, 'swapped_original': None,
        'swap_positions': None, 'planner_output': planner_output,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/eval.yaml")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    name = cfg["model_name"]

    print(f"\n{'='*60}")
    print(f"  model  : {name}")
    print(f"  domain : {cfg['_domain']}")
    print(f"  output : {cfg['_eval_dir']}")
    print(f"{'='*60}\n")

    if not os.path.exists(cfg["_domain"]):
        sys.exit(f"CRITICAL: domain.pddl not found at {cfg['_domain']}\n"
                 f"Run learn_rules first for model '{name}'.")

    os.makedirs(cfg["_pddl_dir"], exist_ok=True)
    os.makedirs(cfg["_eval_dir"], exist_ok=True)

    # ── 1. Load model + train cfg via load_ckpt ───────────────────────────────
    print(f"--> Loading model: {name}")
    model, train_cfg = load_ckpt(name, tag="best")
    cfg["_train_cfg"] = train_cfg          # stash for rel_dim lookups
    _load_data.init(train_cfg)
    set_seed(train_cfg["seed"])
    model.freeze(deterministic=True)
    device = next(model.parameters()).device

    rel_dim = train_cfg["model"]["symbol_size"]
    obj_dim = train_cfg["model"]["obj_symbol_size"]

    # ── 2. Collapse symbols via symbol_semantics ──────────────────────────────
    print("--> Computing collapse symbols...")
    collapse_syms = get_collapse_symbols(model, cfg)
    print(f"  collapse_symbols: {collapse_syms}")

    # req_sym_list is empty for the shortest/tallest task —
    # no occlude/inside goals, just height
    req_sym_list = []

    # ── 3. Height analysis (in memory) ────────────────────────────────────────
    target_min  = cfg["eval"]["target_min_level"]
    target_max  = cfg["eval"]["target_max_level"]
    z_thr       = cfg["eval"]["z_level_threshold"]
    dry_run     = cfg["eval"]["dry_run"]
    one_shot_on = cfg["one_shot"]["enabled"]
    dataset_csv = cfg["data"]["dataset_csv"]

    print("--> Running height analysis (in memory)...")
    df_details, df_summary = compute_height_analysis(dataset_csv, z_thr)
    df_details['collapsed'] = df_details['collapsed'].astype(bool)

    # ── 4. pair → original lookup ─────────────────────────────────────────────
    df_dataset       = load_csv(dataset_csv)
    pair_to_original = build_pair_to_original(df_dataset)
    print(f"  pair→original: {len(pair_to_original)} entries\n")

    # ── 5. Filter to target groups ────────────────────────────────────────────
    targets = df_summary[
        (df_summary['min_levels'] == target_min) &
        (df_summary['max_levels'] == target_max)
    ].reset_index(drop=True)

    print(f"Groups with min={target_min} & max={target_max}: {len(targets)}")
    if targets.empty:
        print("Nothing to run.")
        return

    # ── 6. Pre-compute object symbols for all unique objects (once) ───────────
    print("\n--> Pre-computing object symbols for all unique objects (once)...")
    all_unique_pairs = set()
    for _, grow in targets.iterrows():
        try:
            all_unique_pairs.update(ast.literal_eval(grow['object_group']))
        except Exception:
            pass

    unique_pair_list  = sorted(all_unique_pairs)
    unique_obj_inputs = [pair_to_original[k] for k in unique_pair_list
                         if k in pair_to_original]
    all_symbols       = get_obj_symbols(unique_obj_inputs, model, cfg)
    print(f"  Got symbols for {len(all_symbols)} unique objects")
    print(" all_symbols:", all_symbols)
    symbol_cache = {}
    for key, sym in zip(unique_pair_list, all_symbols):
        symbol_cache[key] = sym
        print(f"  cached: {key} → {sym}")
    print(f"  Symbol cache: {len(symbol_cache)} entries\n")

    # ── 7. One-shot setup (once) ──────────────────────────────────────────────
    if one_shot_on:
        match_cache          = load_match_cache(cfg)
        matched_to_one_shots = build_one_shot_maps_multi(cfg, match_cache)
        n_swaps = sum(len(v) for v in matched_to_one_shots.values())
        if n_swaps == 0:
            sys.exit("CRITICAL: one_shot.enabled=true but no valid matches found.")
        print(f"  {n_swaps} one-shot swap(s) ready\n")
    else:
        matched_to_one_shots = {}
        print("  One-shot disabled\n")

    results = []

    # ── 8. Main loop ──────────────────────────────────────────────────────────
    for i, grow in targets.iterrows():
        group_label = grow['object_group']
        n_objects   = int(grow['n_objects'])

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(targets)}] {group_label}")

        to_be_used, pair_order = group_str_to_object_list(group_label, pair_to_original)
        if to_be_used is None:
            for h in [target_min, target_max]:
                results.append(_empty_row(group_label, n_objects, h, 'SKIP_PARSE_ERROR'))
            continue

        # Object symbols for THIS group (ordered by pair_order)
        obj_symbols_for_group = [symbol_cache[k] for k in pair_order
                                  if k in symbol_cache]
        if len(obj_symbols_for_group) != len(pair_order):
            print(f"  [WARN] Symbol cache incomplete for this group — skipping")
            for h in [target_min, target_max]:
                results.append(_empty_row(group_label, n_objects, h, 'SKIP_SYMBOL_ERROR'))
            continue

        print("  PDDL mapping: " + ", ".join(
            f"obj{k}={v}" for k, v in enumerate(pair_order)))

        for target_height in [target_min, target_max]:
            print(f"\n  ── H{target_height} run ──")
            known_seqs = build_known_sequences(df_details, group_label, target_height)
            print(f"  Known sequences at H{target_height}: {len(known_seqs)}")

            # 1. Generate problem.pddl
            try:
                construct_problem_pddl(
                    obj_symbols_for_group, collapse_syms, req_sym_list,
                    rel_dim, obj_dim, target_height, cfg["_pddl_dir"]
                )
            except Exception as e:
                print(f"  [ERROR] PDDL gen failed: {e}")
                results.append(_empty_row(group_label, n_objects, target_height,
                                          f'PDDL_GEN_ERROR: {e}'))
                continue

            if dry_run:
                shutil.copy(
                    os.path.join(cfg["_pddl_dir"], "problem.pddl"),
                    os.path.join(cfg["_pddl_dir"],
                                 f"problem_H{target_height}_dry_run.pddl"))
                print(f"  [DRY_RUN] saved")
                if target_height == target_max:
                    print("\n[DRY_RUN] Done.")
                continue

            # 2. Run planner
            status, plan_lines, full_output = run_planner(cfg)
            print(f"  Planner: {status}")
            save_pddl_copy(cfg, i, target_height, status)

            plan_str = ''; matched = False
            sim_levels = sim_success = sim_collapsed = image_path = None
            sim_success   = False
            sim_collapsed = None

            if status == "SUCCESS" and plan_lines:
                plan_order_result = extract_order_from_plan(plan_lines, pair_order)
                if plan_order_result:
                    plan_str = " > ".join(plan_order_result)
                    matched  = plan_order_result in known_seqs
                    print(f"  Plan order : {plan_str}")
                    print(f"  Matched known: {matched}")

                    # 3. Simulate
                    sim_levels, sim_success, sim_collapsed, sim_csv = run_simulation(
                        plan_order_result, pair_to_original, cfg, i, target_height)

                    # 4. Image
                    rgb_paths  = get_all_step_rgb_paths(sim_csv) if sim_csv else []
                    image_path = save_result_image(
                        cfg, group_label, i, target_height,
                        plan_str, matched, sim_levels, sim_success,
                        sim_collapsed, status, rgb_paths)
                    print(f"  [IMG] → {image_path}")

                    # 5. One-shot
                    if one_shot_on:
                        os_rows = run_one_shot_variants(
                            cfg, plan_order_result, pair_to_original,
                            matched_to_one_shots, group_label, n_objects,
                            i, target_height, plan_str, matched)
                        results.extend(os_rows)
            else:
                image_path = save_result_image(
                    cfg, group_label, i, target_height,
                    '', False, None, False, None, status, [])

            results.append({
                'group': group_label, 'n_objects': n_objects,
                'target_height': target_height, 'status': status,
                'plan_order': plan_str, 'matched_known_sequence': matched,
                'sim_levels': sim_levels, 'sim_success': sim_success,
                'sim_collapsed': sim_collapsed, 'image_path': image_path,
                'is_one_shot': False, 'one_shot_object': None,
                'swapped_original': None, 'swap_positions': None,
                'planner_output': full_output[:600],
            })

        if dry_run:
            break

    # ── 9. Save + summary ─────────────────────────────────────────────────────
    if dry_run or not results:
        print("\n[DRY_RUN] No results to summarise.")
        return

    df_results = pd.DataFrame(results)
    df_results.to_csv(cfg["_results"], index=False)

    SEP = '=' * 60
    print(f"\n{SEP}\nFINAL SUMMARY\n{SEP}")

    df_base = df_results[~df_results['is_one_shot']]
    for h in [target_min, target_max]:
        sub           = df_base[df_base['target_height'] == h]
        n_ok          = (sub['status'] == 'SUCCESS').sum()
        n_ran         = sub['sim_levels'].notna().sum()
        n_sim_ok      = sub['sim_success'].sum()
        n_coll        = sub['sim_collapsed'].eq(True).sum()
        pct           = ('%.0f' % (100*n_sim_ok/n_ran)) if n_ran else '—'
        print(f"\n  H{h}:")
        print(f"    Plan found       : {n_ok}/{len(sub)}")
        print(f"    Matched known    : {sub['matched_known_sequence'].sum()}/{n_ok}")
        print(f"    Sim ran          : {n_ran}   collapsed={n_coll}")
        print(f"    Sim correct      : {n_sim_ok}/{n_ran}  ({pct}%)")

    df_os = df_results[df_results['is_one_shot'] == True]
    if not df_os.empty:
        print(f"\n── ONE-SHOT ({len(df_os)} runs) ──")
        for osl in sorted(df_os['one_shot_object'].dropna().unique()):
            sub = df_os[df_os['one_shot_object'] == osl]
            for h in [target_min, target_max]:
                h_sub = sub[sub['target_height'] == h]
                if h_sub.empty: continue
                n_ran = h_sub['sim_levels'].notna().sum()
                n_ok  = h_sub['sim_success'].sum()
                pct   = ('%.0f' % (100*n_ok/n_ran)) if n_ran else '—'
                print(f"  [{osl}] H{h}: {n_ok}/{n_ran} ({pct}%)")

    print(f"\nSaved → {cfg['_results']}")
    print(f"Images → {cfg['_img_dir']}")


if __name__ == '__main__':
    main()