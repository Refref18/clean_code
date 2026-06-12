"""
plan_task_clean.py
==================
Clean version of plan_task.py in the same style as plan_height.py:

  • no PDDL.problem_class.ProblemGenerator
  • no PDDL.problem.py helper calls
  • object symbols are computed directly from depth images via the loaded model
  • collapse / task relation symbols are computed directly via symbol_semantics
  • problem.pddl is written in this file

Usage:
    python plan_task_clean.py -c configs/eval.yaml

eval:
  task: inside                  # inside or occlude
  n_baseline_scenarios: 5
  seed: 42
  task_output_dim: 0
  feasibility_tries: 30
"""

import argparse
import ast
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import load_ckpt
from symbol_semantics import get_semantic_symbols, get_symbol_map
from data_collection_direct.exp import get_experiment
import load_data as _load_data
from load_data import set_seed as project_set_seed

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ONE_SHOT_EXTRA = [
    ("cup",      "objects/cup12_w32_h32.urdf"),
    ("cylinder", "0.14"),
    ("sphere",   "0.12"),
    ("box",      "0.12"),
    ("cup",      "one_shot_objects/smooth_cone.urdf"),
    ("cup",      "one_shot_objects/plus_w14_h10.urdf"),
    ("cup",      "one_shot_objects/cup12_w16_h16.urdf"),
]

DEFAULT_TASK_OUTPUT_DIMS = {"inside": 0, "occlude": 1}

DEFAULT_TUPLE_LIMITS = {
    ("inverted_cup", "inv_cup_w20_h20_thinner.urdf"): 1,
    ("inverted_cup", "inv_cup_w18_h18_thinner.urdf"): 1,
    ("inverted_cup", "inv_cup_w16_h16_thinner.urdf"): 1,
    ("cup", "cup_w20_h20.urdf"): 1,
    ("cup", "cup_w18_h18.urdf"): 1,
    ("cup", "cup_w16_h16.urdf"): 1,
    ("sphere", "0.14"): 5,
    ("box", "0.14"): 1,
}

DEFAULT_TUPLE_PAIRS = [
    (("cup", "cup_w20_h20.urdf"), ("inverted_cup", "inv_cup_w20_h20_thinner.urdf")),
    (("cup", "cup_w18_h18.urdf"), ("inverted_cup", "inv_cup_w18_h18_thinner.urdf")),
    (("cup", "cup_w16_h16.urdf"), ("inverted_cup", "inv_cup_w16_h16_thinner.urdf")),
]

KNOWN_SIZES = [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13,
               0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]


def _require_cfg(cfg, dotted_key):
    cur = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing required config key: {dotted_key}")
        cur = cur[part]
    return cur


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    name = cfg["model_name"]
    save_root = os.path.join("save", name)
    eval_dir = os.path.join(save_root, "eval_task")

    _require_cfg(cfg, "data.dataset_csv")
    _require_cfg(cfg, "data.depth_image_dir")
    _require_cfg(cfg, "planner.fast_downward")
    _require_cfg(cfg, "planner.search")
    _require_cfg(cfg, "eval.task")

    task = cfg["eval"]["task"]
    min_h = cfg["eval"].get("min_objects", "any")
    max_h = cfg["eval"].get("max_objects", "any")
    task_dir_name = f"{task}_H{min_h}_H{max_h}_new"
    one_shot_enabled = bool(cfg.get("one_shot", {}).get("enabled", False))

    cfg.setdefault("one_shot", {})
    cfg["one_shot"].setdefault("enabled", one_shot_enabled)
    cfg["one_shot"].setdefault("match_cache_path", "match_results_cache.json")
    cfg["one_shot"].setdefault("n_scenarios_per_one_shot", 20 if one_shot_enabled else 0)
    cfg["one_shot"].setdefault("in_pair_ratio", 0.0)
    cfg["one_shot"].setdefault("objects", DEFAULT_ONE_SHOT_EXTRA)

    cfg["_domain"]    = os.path.join(save_root, "domain.pddl")
    cfg["_pddl_root"] = os.path.join(save_root, "PDDL_FILES")
    cfg["_pddl_dir"]  = os.path.join(cfg["_pddl_root"], task_dir_name)
    cfg["_plan_file"] = os.path.join(cfg["_pddl_dir"], "sas_plan")
    cfg["_eval_dir"]  = eval_dir
    cfg["_exp_dir"]   = os.path.join(eval_dir, f"{task_dir_name}_sim_experiments")
    cfg["_img_dir"]   = os.path.join(eval_dir, f"{task_dir_name}_images")
    cfg["_results"]   = os.path.join(eval_dir, f"eval_results_{task_dir_name}.csv")
    cfg["_recovery"]  = os.path.join(eval_dir, f"{task_dir_name}_no_plan_visuals")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def load_csv(path):
    for enc in ["utf-8", "cp1254", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    sys.exit(f"CRITICAL: Cannot read {path}")


def situation_matches_task(situation, task):
    s = str(situation).strip().lower()
    t = str(task).strip().lower()
    return s == t or t in s


def obj_to_canonical(obj_type, obj_size_or_path):
    try:
        float(obj_size_or_path)
        return (str(obj_type), str(obj_size_or_path))
    except (ValueError, TypeError):
        clean = str(obj_size_or_path).split("/")[-1].replace(".urdf", "")
        return (str(obj_type), clean)


def extract_object_name(path):
    if pd.isna(path):
        return "unknown"
    filename = os.path.basename(str(path)).replace(".urdf", "")
    parts = filename.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else parts[0]


def clean_obj_label(obj):
    if obj is None:
        return "None"
    return f"{obj_to_canonical(*obj)[0]}_{obj_to_canonical(*obj)[1]}"


# ─────────────────────────────────────────────────────────────────────────────
# Object symbol computation
# ─────────────────────────────────────────────────────────────────────────────

def _clean_size(obj_size_or_path):
    try:
        val = float(obj_size_or_path)
        return str(min(KNOWN_SIZES, key=lambda x: abs(x - val)))
    except (ValueError, TypeError):
        return str(obj_size_or_path).split("/")[-1].replace(".urdf", "")


def load_depth_image(obj_type, obj_size_or_path, depth_dir, device):
    clean = _clean_size(obj_size_or_path)
    base = f"{obj_type}_{clean}"
    upper_p = os.path.join(depth_dir, f"upper_{base}_new.npy")
    lower_p = os.path.join(depth_dir, f"lower_{base}_new.npy")
    if os.path.exists(upper_p) and os.path.exists(lower_p):
        u = torch.from_numpy(np.load(upper_p)).unsqueeze(0).float()
        l = torch.from_numpy(np.load(lower_p)).unsqueeze(0).float()
        return torch.cat([u, l], dim=0).to(device)
    print(f"  [WARN] Depth images missing for {base} — using zeros")
    return torch.zeros(2, 64, 64, device=device)


def get_obj_symbols(ordered_objects, model, cfg):
    """
    Compute object symbols exactly like PLAN_HEIGHT.py: one object at a time,
    wrapped as a single-node PyG graph and passed through a single-item
    DataLoader before image encoding + centroid lookup.

    This avoids the wrong-symbol issue caused by calling model.get_object_symbols
    on a stacked tensor of several objects, while also avoiding the full
    preprocessing path that may open PyBullet clients.
    """
    import torch.nn.functional as F

    depth_dir = cfg["data"]["depth_image_dir"]
    device = next(model.parameters()).device

    model.eval()
    model.gs_obj_layer.hard = True
    model.gs_obj_layer.deterministic = True

    results = []
    with torch.no_grad():
        for otype, opath in ordered_objects:
            # Keep image loading on CPU first, then let the DataLoader create the
            # same single-graph batch structure used in PLAN_HEIGHT.py.
            img = load_depth_image(otype, opath, depth_dir, torch.device("cpu"))
            single = Data(
                x=img.unsqueeze(0),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 5), dtype=torch.float),
            )
            loader = DataLoader([single], batch_size=1, shuffle=False, num_workers=0)
            batch = next(iter(loader))

            x = batch.x.float().to(device)
            feats = model.image_encoder(x)
            sim = F.normalize(feats, dim=-1) @ F.normalize(model.cluster_centroids, dim=-1).T
            bits = model.cluster_codes[sim.argmax(dim=-1)]
            results.append(bits.squeeze(0).cpu().int().tolist())

    return results

def build_object_symbol_cache(objects, model, cfg):
    unique = []
    seen = set()
    for obj in objects:
        canon = obj_to_canonical(*obj)
        if canon not in seen:
            seen.add(canon)
            unique.append(obj)
    if not unique:
        return {}
    syms = get_obj_symbols(unique, model, cfg)
    cache = {}
    for obj, sym in zip(unique, syms):
        cache[obj_to_canonical(*obj)] = sym
        print(f"  cached object symbol: {obj_to_canonical(*obj)} -> {sym}")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Relation / task symbol computation
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_task_output_dim(cfg, task):
    ev = cfg.get("eval", {})
    candidates = [
        ev.get("task_output_dim"),
        ev.get(f"{task}_output_dim"),
        cfg.get("task_output_dims", {}).get(task),
        cfg.get("tasks", {}).get(task, {}).get("output_dim"),
    ]
    for c in candidates:
        if c is not None:
            return int(c)
    fallback = DEFAULT_TASK_OUTPUT_DIMS.get(task)
    if fallback is None:
        raise ValueError(f"No output dim configured for task={task!r}. Set eval.task_output_dim.")
    print(f"  [WARN] eval.task_output_dim is not set; using fallback {fallback} for task={task}")
    return fallback


def get_collapse_symbols(model, cfg):
    train_cfg = cfg["_train_cfg"]
    rel_dim = train_cfg["model"]["symbol_size"]
    out_dim = 4
    device = next(model.parameters()).device
    collapse_syms, _, _ = get_semantic_symbols(
        model, sym_size=rel_dim, out_dim=out_dim, collapse_threshold=0.5, device=device)
    return [list(s) for s in collapse_syms]


def get_task_relation_symbols(model, cfg, task):
    train_cfg = cfg["_train_cfg"]
    rel_dim = train_cfg["model"]["symbol_size"]
    collapse_out_dim = 4
    device = next(model.parameters()).device
    eps = float(cfg.get("eval", {}).get("width_epsilon", 0.0))

    _, inserted_syms, _ = get_semantic_symbols(
        model, sym_size=rel_dim, out_dim=collapse_out_dim, collapse_threshold=0.5, device=device)
    inserted_syms = [list(s) for s in inserted_syms]

    if not inserted_syms:
        print(f"  [WARN] No inserted symbols found; task={task} has no candidates")
        return []

    symbol_map = get_symbol_map(model, sym_size=rel_dim, out_dim=collapse_out_dim, device=device)
    scored = []
    for bits in inserted_syms:
        key = tuple(float(b) for b in bits)
        effect = symbol_map[key]
        min_x = effect[0].item()
        max_x = effect[2].item()
        width_delta = max_x - min_x
        scored.append((bits, width_delta, min_x, max_x))
    scored = sorted(scored, key=lambda x: x[1])

    print("=" * 60)
    print(f"  TASK SYMBOL CLASSIFICATION ({task})")
    print("  candidates = inserted symbols split by decoded width")
    print("  width_delta = decoded MaxX - decoded MinX")
    for bits, width_delta, min_x, max_x in scored:
        print(f"    {bits}  width_delta={width_delta:.4f}  min_x={min_x:.4f}  max_x={max_x:.4f}")

    task_l = str(task).strip().lower()
    if task_l == "inside":
        selected = [bits for bits, width_delta, _, _ in scored if width_delta < -eps]
        if not selected:
            selected = [scored[0][0]]
            print("  [WARN] No negative-width inserted symbol; using narrowest symbol as inside.")
    elif task_l == "occlude":
        selected = [bits for bits, width_delta, _, _ in scored if width_delta > eps]
        if not selected:
            selected = [scored[-1][0]]
            print("  [WARN] No positive-width inserted symbol; using widest symbol as occlude.")
    else:
        raise ValueError("Task-specific width split is implemented only for inside/occlude.")

    print(f"  selected {task_l} symbols: {selected}")
    print("=" * 60)
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Dataset / task extraction
# ─────────────────────────────────────────────────────────────────────────────

def find_qualifying_towers(df, task):
    tower_steps = defaultdict(list)
    tower_has_task = set()
    for _, row in df.iterrows():
        tid = row["id"]
        collapse_val = row.get("collapse", False)
        try:
            if float(collapse_val) >= 0.5:
                continue
        except (TypeError, ValueError):
            if str(collapse_val).strip().lower() in ("true", "1", "yes"):
                continue
        tower_steps[tid].append(row.to_dict())
        diff_raw = row.get("bounding_box_differences")
        if pd.isna(diff_raw):
            continue
        try:
            bbox_diff = ast.literal_eval(str(diff_raw))
        except Exception:
            continue
        for entry in bbox_diff.values():
            if isinstance(entry, dict) and situation_matches_task(entry.get("situation", ""), task):
                tower_has_task.add(tid)
                break
    return {
        tid: sorted(tower_steps[tid], key=lambda r: int(r["step"]))
        for tid in tower_has_task
        if tower_steps[tid]
    }


def build_object_list_from_steps(steps):
    order = []
    for row in sorted(steps, key=lambda r: int(r["step"])):
        obj_type = str(row.get("object_type", ""))
        raw_val = row.get("object_size_or_path", "")
        order.append((obj_type, raw_val))
    return order


def get_task_pairs_from_tower(steps, task):
    placement_order = {
        int(row["step"]) - 1: (str(row.get("object_type", "")),
                               str(row.get("object_size_or_path", "")))
        for row in steps
    }
    pairs = []
    for row in steps:
        step = int(row["step"])
        if step == 1:
            continue
        diff_raw = row.get("bounding_box_differences")
        if pd.isna(diff_raw):
            continue
        try:
            bbox_diff = ast.literal_eval(str(diff_raw))
        except Exception:
            continue
        new_idx = step - 1
        for k, entry in bbox_diff.items():
            if not isinstance(entry, dict):
                continue
            sit = str(entry.get("situation", "")).strip().lower()
            if not situation_matches_task(sit, task):
                continue
            existing_idx = int(k)
            existing_obj = placement_order.get(existing_idx)
            new_obj = placement_order.get(new_idx)
            if task == "inside" and existing_obj and existing_obj[0].lower() != "cup":
                continue
            pairs.append({
                "new_obj_idx": new_idx,
                "existing_obj_idx": existing_idx,
                "new_obj": new_obj,
                "existing_obj": existing_obj,
                "step": step,
                "situation": sit,
            })
    return pairs


def build_task_groups(qualifying, task):
    task_groups = defaultdict(list)
    for tid, steps in qualifying.items():
        obj_list = build_object_list_from_steps(steps)
        obj_pool = tuple(sorted(obj_to_canonical(*o) for o in obj_list))
        expected_pairs = get_task_pairs_from_tower(steps, task)
        pair_fingerprint = tuple(sorted(
            (obj_to_canonical(*p["new_obj"]), obj_to_canonical(*p["existing_obj"]))
            for p in expected_pairs
        ))
        task_groups[(obj_pool, pair_fingerprint)].append(tid)
    return task_groups


def filter_task_groups(task_groups, allowed_ids):
    allowed = set(allowed_ids)
    filtered = {}
    for key, tids in task_groups.items():
        kept = [t for t in tids if t in allowed]
        if kept:
            filtered[key] = list(kept)
    return filtered


def sample_diverse_towers(task_groups_copy, n):
    sampled = []
    group_keys = list(task_groups_copy.keys())
    random.shuffle(group_keys)
    idx = 0
    while len(sampled) < n and group_keys:
        key = group_keys[idx % len(group_keys)]
        if task_groups_copy[key]:
            sampled.append(task_groups_copy[key].pop(random.randrange(len(task_groups_copy[key]))))
        else:
            group_keys.pop(idx % len(group_keys))
            if not group_keys:
                break
        idx += 1
    return sampled


def get_pair_tower_ids(qualifying, matched_canon, task):
    pair_ids, nonpair_ids = [], []
    for tid, steps in qualifying.items():
        expected = get_task_pairs_from_tower(steps, task)
        pair_objs = set()
        for ep in expected:
            for k in ("new_obj", "existing_obj"):
                obj = ep.get(k)
                if obj:
                    pair_objs.add(obj_to_canonical(*obj))
        if matched_canon in pair_objs:
            pair_ids.append(tid)
        else:
            nonpair_ids.append(tid)
    return pair_ids, nonpair_ids


# ─────────────────────────────────────────────────────────────────────────────
# Planning case construction
# ─────────────────────────────────────────────────────────────────────────────

def _case_group_key(object_list, expected_pairs):
    obj_pool = tuple(sorted(obj_to_canonical(*o) for o in object_list))
    pair_fingerprint = tuple(sorted(
        (obj_to_canonical(*p["new_obj"]), obj_to_canonical(*p["existing_obj"]))
        for p in expected_pairs
        if p.get("new_obj") is not None and p.get("existing_obj") is not None
    ))
    return obj_pool, pair_fingerprint


def make_real_cases(qualifying, task):
    cases = []
    for tid, steps in qualifying.items():
        object_list = build_object_list_from_steps(steps)
        expected_pairs = get_task_pairs_from_tower(steps, task)
        if not expected_pairs:
            continue
        cases.append({
            "case_id": str(tid),
            "source": "real",
            "steps": steps,
            "object_list": object_list,
            "expected_pairs": expected_pairs,
            "group_key": _case_group_key(object_list, expected_pairs),
        })
    return cases


def sample_diverse_cases(cases, n):
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["group_key"]].append(case)
    sampled = []
    group_keys = list(grouped.keys())
    random.shuffle(group_keys)
    idx = 0
    while len(sampled) < n and group_keys:
        key = group_keys[idx % len(group_keys)]
        if grouped[key]:
            sampled.append(grouped[key].pop(random.randrange(len(grouped[key]))))
        else:
            group_keys.pop(idx % len(group_keys))
            if not group_keys:
                break
        idx += 1
    return sampled


def _shift_pair_indices(pair, offset):
    q = dict(pair)
    q["new_obj_idx"] = int(q["new_obj_idx"]) + offset
    q["existing_obj_idx"] = int(q["existing_obj_idx"]) + offset
    q["step"] = int(q["step"]) + offset
    return q


def build_base_fragments(qualifying, task, tuple_limits, tuple_pairs):
    """
    Build base fragments for synthetic case construction.

    For each tower and each task pair within it, create a PREFIX fragment
    ending exactly at the step where the inside/occlude relation occurs.
    This guarantees the task pair objects are always adjacent in the sequence
    (container first, inserted object immediately after), which is physically
    required for inside to work.
    """
    fragments = []
    for tid, steps in qualifying.items():
        full_obj_list = build_object_list_from_steps(steps)
        expected_pairs = get_task_pairs_from_tower(steps, task)
        if not full_obj_list or not expected_pairs:
            continue
        for pair in expected_pairs:
            step = int(pair["step"])
            prefix_objects = full_obj_list[:step]
            if not prefix_objects:
                continue
            if not obeys_tuple_constraints(prefix_objects, tuple_limits, tuple_pairs):
                continue
            prefix_pairs = [
                dict(p) for p in expected_pairs
                if int(p["new_obj_idx"]) < step and int(p["existing_obj_idx"]) < step
            ]
            if not prefix_pairs:
                continue
            fragments.append({
                "tower_id": str(tid),
                "objects": prefix_objects,
                "pairs": prefix_pairs,
            })
    return fragments


def _deduplicate_expected_pairs(expected_pairs):
    """
    Keep exactly ONE physically valid task pair from a synthetic case.
    An object can only be inside ONE container at a time.
    """
    for ep in expected_pairs:
        new_obj = ep.get("new_obj")
        existing_obj = ep.get("existing_obj")
        if new_obj is None or existing_obj is None:
            continue
        new_canon = obj_to_canonical(*new_obj)
        ex_canon = obj_to_canonical(*existing_obj)
        print(f"    [dedup] keeping single pair: {new_canon} -> {ex_canon}")
        return [ep]
    return []


def build_synthetic_case(case_idx, base_fragments, target_n,
                         tuple_limits, tuple_pairs, max_attempts=2000):
    if not base_fragments:
        return None
    full_fragments = [f for f in base_fragments if len(f["objects"]) <= target_n]
    if not full_fragments:
        return None

    for _ in range(max_attempts):
        base_frag = random.choice(full_fragments)
        objects = list(base_frag["objects"])
        expected_pairs = [dict(p) for p in base_frag["pairs"]]
        sources = [f"{base_frag['tower_id']}[:FULL]"]

        if not obeys_tuple_constraints(objects, tuple_limits, tuple_pairs):
            continue

        while len(objects) < target_n:
            remaining = target_n - len(objects)
            frag = random.choice(base_fragments)
            take = min(remaining, len(frag["objects"]))
            appended = False
            for k in range(take, 0, -1):
                candidate_objects = objects + frag["objects"][:k]
                if not obeys_tuple_constraints(candidate_objects, tuple_limits, tuple_pairs):
                    continue
                offset = len(objects)
                for pair in frag["pairs"]:
                    if int(pair["new_obj_idx"]) < k and int(pair["existing_obj_idx"]) < k:
                        expected_pairs.append(_shift_pair_indices(pair, offset))
                objects = candidate_objects
                sources.append(f"{frag['tower_id']}[:{k}]")
                appended = True
                break
            if not appended:
                break

        if len(objects) == target_n and expected_pairs:
            expected_pairs = _deduplicate_expected_pairs(expected_pairs)
            if not expected_pairs:
                continue
            return {
                "case_id": f"synthetic_{case_idx:04d}",
                "source": "synthetic",
                "steps": None,
                "object_list": objects,
                "expected_pairs": expected_pairs,
                "sources": sources,
                "group_key": _case_group_key(objects, expected_pairs),
            }
    return None


def build_synthetic_cases(base_fragments, target_n, n_cases,
                          tuple_limits, tuple_pairs, max_attempts_per_case=2000):
    """
    Build synthetic object groups. Validation happens lazily at planning time.
    """
    cases = []
    seen = set()
    attempts = 0
    max_total_attempts = max(n_cases * 50, 1000)
    while len(cases) < n_cases and attempts < max_total_attempts:
        attempts += 1
        case = build_synthetic_case(
            len(cases), base_fragments, target_n,
            tuple_limits, tuple_pairs, max_attempts=max_attempts_per_case)
        if case is None:
            continue
        signature = tuple(obj_to_canonical(*o) for o in case["object_list"])
        if signature in seen and attempts < max_total_attempts // 2:
            continue
        seen.add(signature)
        cases.append(case)
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# Feasibility check — lazy, only called on NO_PLAN
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup_trial(csv_path):
    if csv_path and os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except OSError:
            pass


def is_achievable_order(object_list, expected_pairs, task, cfg,
                        n_tries=30, bias_pairs=True):
    """
    Tries up to n_tries random orderings via physics simulation.
    Threshold is 100%: partial achievement means the case is still
    over-constrained and should be discarded.
    """
    feasibility_dir = os.path.join(cfg["_exp_dir"], "_feasibility")
    os.makedirs(feasibility_dir, exist_ok=True)
    n_expected = len(expected_pairs)

    pair_canons = set()
    if bias_pairs:
        for ep in expected_pairs:
            for key in ("new_obj", "existing_obj"):
                obj = ep.get(key)
                if obj is not None:
                    pair_canons.add(obj_to_canonical(*obj))

    def _biased_shuffle(obj_list):
        non_pair = [o for o in obj_list if obj_to_canonical(*o) not in pair_canons]
        pair_objs = [o for o in obj_list if obj_to_canonical(*o) in pair_canons]
        random.shuffle(non_pair)
        random.shuffle(pair_objs)
        if task == "inside" and len(pair_objs) >= 2:
            cups = [o for o in pair_objs if o[0].lower() in ("cup", "inverted_cup")]
            others = [o for o in pair_objs if o not in cups]
            pair_objs = cups + others
        return non_pair + pair_objs

    tried_signatures = set()
    n_actual_tries = 0
    best_pct_seen = 0.0

    for attempt in range(n_tries * 3):
        if n_actual_tries >= n_tries:
            break
        if bias_pairs and attempt < n_tries // 2:
            candidate = _biased_shuffle(list(object_list))
        else:
            candidate = list(object_list)
            random.shuffle(candidate)
        sig = tuple(obj_to_canonical(*o) for o in candidate)
        if sig in tried_signatures:
            continue
        tried_signatures.add(sig)
        n_actual_tries += 1

        trial_label = f"feasibility_{uuid.uuid4().hex[:8]}"
        trial_csv = os.path.join(feasibility_dir, f"{trial_label}.csv")
        try:
            get_experiment(candidate, feasibility_dir + "/", trial_label)
        except Exception as e:
            print(f"    [feasibility] simulation error on attempt {n_actual_tries}: {e}")
            _cleanup_trial(trial_csv)
            continue

        actual_csv, collapsed = truncate_csv_at_collapse(trial_csv)
        if collapsed:
            _cleanup_trial(actual_csv)
            continue

        found_pairs = check_task_in_result(actual_csv, task, candidate, {})
        _, n_matched, pct, _, _ = compute_success(expected_pairs, found_pairs)
        _cleanup_trial(actual_csv)

        if pct > best_pct_seen:
            best_pct_seen = pct
        if pct == 100.0:
            print(f"    [feasibility] ✓ fully achievable after {n_actual_tries} tries "
                  f"(100%  order={[clean_obj_label(o) for o in candidate]})")
            return True, candidate

    if best_pct_seen > 0:
        print(f"    [feasibility] ✗ best result was {best_pct_seen:.0f}% "
              f"({int(best_pct_seen * n_expected / 100)}/{n_expected} pairs) "
              f"in {n_actual_tries} tries — expected_pairs over-constrained, discarding")
    else:
        print(f"    [feasibility] ✗ no task relation achieved in {n_actual_tries} tries "
              f"— object group invalid, discarding")
    return False, None


# ─────────────────────────────────────────────────────────────────────────────
# PDDL problem construction
# ─────────────────────────────────────────────────────────────────────────────

def _object_sym_to_pddl(obj_symbols, obj_dim, indentation="\t\t"):
    schema = ""
    for obj_num, obj_val in enumerate(obj_symbols):
        name = f"obj{obj_num}"
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
    possibilities = []
    for target in spec_obj_syms:
        indices = [i for i, val in enumerate(all_obj_syms) if val == target]
        possibilities.append(indices)
    if any(len(p) == 0 for p in possibilities):
        return []
    combos = [[]]
    for indices in possibilities:
        combos = [prev + [idx] for prev in combos for idx in indices]
    return [tuple(f"obj{idx}" for idx in combo) for combo in combos]


def _get_spec_possibilities(rel_sym, all_obj_syms, spec_obj_syms,
                             rel_dim, indentation="\t\t\t"):
    pairs = _spec_pairs(all_obj_syms, spec_obj_syms)
    if not pairs:
        return indentation + "(and (all-used) (not (all-used)))\n"
    schema = indentation + "(or\n"
    for n1, n2 in pairs:
        schema += indentation + "\t(and "
        for k, val in enumerate(rel_sym):
            schema += f"(r{k} {n1} {n2}) " if val == 1 else f"(not_r{k} {n1} {n2}) "
        schema += f"(r{rel_dim} {n1} {n2}) )\n"
    schema += indentation + ")\n"
    return schema


def _goal_with_obj_restriction(req_sym_list, all_obj_syms, rel_dim, indentation="\t\t"):
    schema = ""
    for case in req_sym_list:
        schema += indentation + "(or\n"
        for possible_sym in case["pos_rel_symbols"]:
            schema += _get_spec_possibilities(
                possible_sym, all_obj_syms, case["obj_symbols"], rel_dim,
                indentation=indentation + "\t")
        schema += indentation + ")\n"
    return schema


def _goal_any_pair_pddl(n_objs, task_pos_rel_symbols, rel_dim, indentation="\t\t"):
    """
    Mode B goal: ANY ordered pair of objects may satisfy the task relation.
    Produces a single (or ...) over all N*(N-1) ordered pairs for every
    task-positive relation symbol.
    """
    schema = indentation + "(or\n"
    for rel_sym in task_pos_rel_symbols:
        for i in range(n_objs):
            for j in range(n_objs):
                if i == j:
                    continue
                n1, n2 = f"obj{i}", f"obj{j}"
                schema += indentation + "\t(and "
                for k, val in enumerate(rel_sym):
                    schema += f"(r{k} {n1} {n2}) " if val == 1 else f"(not_r{k} {n1} {n2}) "
                schema += f"(r{rel_dim} {n1} {n2}) )\n"
    schema += indentation + ")\n"
    return schema


def _make_required_goal_cases(expected_pairs, task_pos_rel_symbols, object_symbol_cache):
    req_sym_list = []
    for ep in expected_pairs:
        existing_obj = ep.get("existing_obj")
        new_obj = ep.get("new_obj")
        if existing_obj is None or new_obj is None:
            continue
        ex_sym = object_symbol_cache.get(obj_to_canonical(*existing_obj))
        nw_sym = object_symbol_cache.get(obj_to_canonical(*new_obj))
        if ex_sym is None or nw_sym is None:
            print(f"  [WARN] Missing object symbol for expected pair: {existing_obj}, {new_obj}")
            continue
        req_sym_list.append({
            "pos_rel_symbols": task_pos_rel_symbols,
            "obj_symbols": [ex_sym, nw_sym],
            "debug_pair": (existing_obj, new_obj),
        })
    return req_sym_list


def _usage_goal_for_n(n_objs):
    """
    For towers with fewer than 4 objects the domain uses specific active-count
    predicates instead of the generic all-used predicate.
    """
    if n_objs < 4:
        return f"\t\t(active-count-{n_objs})\n"
    return "\t\t(all-used)\n"


def construct_task_problem_pddl(obj_symbols, collapse_syms, req_sym_list,
                                rel_dim, obj_dim, save_dir, run_label=None):
    """Mode A: specific pair goal."""
    os.makedirs(save_dir, exist_ok=True)
    n_objs = len(obj_symbols)

    problem  = "(define (problem blocks-problem)\n"
    problem += "\t(:domain blocks)\n"
    problem += "\t(:objects\n\t\t"
    problem += " ".join(f"obj{i}" for i in range(n_objs)) + " - object\n\t)\n"
    problem += "\t(:init\n"
    problem += "\t\t(H0)\n"
    problem += "\t\t(active-count-0)\n"
    problem += _object_sym_to_pddl(obj_symbols, obj_dim)
    problem += _initial_relation_sym_to_pddl(n_objs, rel_dim)
    problem += "\t)\n"
    problem += "\t(:goal (and\n"
    problem += _usage_goal_for_n(n_objs)
    problem += "\t\t(not (active-count-collapse))\n"
    problem += _goal_with_obj_restriction(req_sym_list, obj_symbols, rel_dim)
    problem += "\t))\n)"

    out_path = os.path.join(save_dir, "problem.pddl")
    with open(out_path, "w") as f:
        f.write(problem)
    print(f"  -> {out_path} written  ({len(req_sym_list)} required task pair goal(s))")
    return out_path


def construct_task_problem_pddl_any(obj_symbols, collapse_syms, task_pos_rel_symbols,
                                    rel_dim, obj_dim, save_dir, run_label=None):
    """
    Mode B: goal requires ANY inside/occlude relation between any two objects.
    Identical init to Mode A; goal is a single (or ...) over all N*(N-1) pairs.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_objs = len(obj_symbols)

    problem  = "(define (problem blocks-problem)\n"
    problem += "\t(:domain blocks)\n"
    problem += "\t(:objects\n\t\t"
    problem += " ".join(f"obj{i}" for i in range(n_objs)) + " - object\n\t)\n"
    problem += "\t(:init\n"
    problem += "\t\t(H0)\n"
    problem += "\t\t(active-count-0)\n"
    problem += _object_sym_to_pddl(obj_symbols, obj_dim)
    problem += _initial_relation_sym_to_pddl(n_objs, rel_dim)
    problem += "\t)\n"
    problem += "\t(:goal (and\n"
    problem += _usage_goal_for_n(n_objs)
    problem += "\t\t(not (active-count-collapse))\n"
    problem += _goal_any_pair_pddl(n_objs, task_pos_rel_symbols, rel_dim)
    problem += "\t))\n)"

    out_path = os.path.join(save_dir, "problem_any.pddl")
    with open(out_path, "w") as f:
        f.write(problem)
    n_pairs = n_objs * (n_objs - 1)
    print(f"  -> {out_path} written  (any-pair goal, {n_pairs} ordered pairs)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────────────────────

def run_planner(problem_path, cfg):
    plan_file = cfg["_plan_file"]
    if os.path.exists(plan_file):
        os.remove(plan_file)

    time_limit = int(cfg["planner"].get("time_limit_sec", 600))
    mem_limit = int(cfg["planner"].get("memory_limit_mb", 8192))

    cmd = [
        sys.executable, cfg["planner"]["fast_downward"],
        "--overall-time-limit", f"{time_limit}s",
        "--overall-memory-limit", f"{mem_limit}M",
        "--plan-file", plan_file,
        cfg["_domain"], problem_path,
        "--search", cfg["planner"]["search"],
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=time_limit + 30)
        stdout = result.stdout + result.stderr
        if result.returncode == 0 and os.path.exists(plan_file):
            with open(plan_file) as f:
                plan_lines = [l.strip() for l in f if l.strip() and not l.startswith(";")]
            return "SUCCESS", plan_lines, stdout
        if result.returncode == 30:
            return "TIMEOUT", [], stdout
        if result.returncode == 12 or "NO PLAN FOUND" in stdout.upper():
            return "NO_PLAN", [], stdout
        return f"ERROR_{result.returncode}", [], stdout
    except subprocess.TimeoutExpired:
        return "OS_TIMEOUT", [], "The operating system killed the process after the time limit."


def extract_order_from_plan(plan_lines, object_list):
    placed = []
    for line in plan_lines:
        line = line.strip().lstrip("(").rstrip(")")
        tokens = line.split()
        if not tokens or not tokens[0].startswith("a_"):
            continue
        obj_tokens = [t for t in tokens[1:] if re.match(r"^obj\d+$", t)]
        if obj_tokens:
            idx = int(obj_tokens[-1].replace("obj", ""))
            if idx < len(object_list):
                placed.append(object_list[idx])
    return placed


def save_planner_artifacts(problem_path, status, cfg, run_label):
    folder = "successful_plans" if status == "SUCCESS" else "no_plan_files"
    dst_dir = os.path.join(cfg["_pddl_dir"], folder)
    os.makedirs(dst_dir, exist_ok=True)
    safe = re.sub(r"[^\w\-]", "_", run_label)
    if os.path.exists(problem_path):
        shutil.copy(problem_path, os.path.join(dst_dir, f"{safe}_{status}.pddl"))
    plan_file = cfg["_plan_file"]
    if os.path.exists(plan_file):
        shutil.copy(plan_file, os.path.join(dst_dir, f"{safe}_{status}.sas_plan"))


# ─────────────────────────────────────────────────────────────────────────────
# Simulation + result checking
# ─────────────────────────────────────────────────────────────────────────────

def _csv_candidates(path):
    return [
        path,
        path.replace(".csv", "worker.csv"),
        os.path.join(os.path.dirname(path),
                     os.path.basename(path).replace(".csv", "") + "worker.csv"),
    ]


def _find_existing_csv(path):
    for p in _csv_candidates(path):
        if os.path.exists(p):
            return p
    return None


def _row_is_collapsed(value):
    try:
        return float(value) >= 0.5
    except Exception:
        return str(value).strip().lower() in ("true", "1", "yes")


def truncate_csv_at_collapse(csv_path):
    actual_csv = _find_existing_csv(csv_path)
    if actual_csv is None:
        return csv_path, True
    try:
        df = load_csv(actual_csv)
    except Exception:
        return actual_csv, True
    if df.empty or "collapse" not in df.columns:
        return actual_csv, False
    keep_rows = []
    collapsed = False
    for _, row in df.iterrows():
        keep_rows.append(row)
        if _row_is_collapsed(row.get("collapse", False)):
            collapsed = True
            break
    pd.DataFrame(keep_rows).to_csv(actual_csv, index=False)
    return actual_csv, collapsed


def check_task_in_result(experiment_csv_path, task, planned_order_actual,
                         one_shot_to_matched):
    actual_csv = _find_existing_csv(experiment_csv_path)
    if actual_csv is None:
        print(f"  [CHECK] Result CSV not found, tried: {_csv_candidates(experiment_csv_path)}")
        return []
    try:
        df = load_csv(actual_csv)
    except Exception:
        return []

    def normalise(obj):
        if obj is None:
            return obj
        canon = obj_to_canonical(*obj)
        matched_canon = one_shot_to_matched.get(canon)
        return matched_canon if matched_canon else obj

    found_pairs = []
    for _, row in df.iterrows():
        step = int(row.get("step", 0))
        diff_raw = row.get("bounding_box_differences")
        if pd.isna(diff_raw):
            continue
        try:
            bbox_diff = ast.literal_eval(str(diff_raw))
        except Exception:
            continue
        new_idx = step - 1
        new_obj_actual = planned_order_actual[new_idx] if new_idx < len(planned_order_actual) else None
        for k, entry in bbox_diff.items():
            if not isinstance(entry, dict):
                continue
            sit = str(entry.get("situation", "")).strip().lower()
            if not situation_matches_task(sit, task):
                continue
            existing_idx = int(k)
            existing_obj_actual = (planned_order_actual[existing_idx]
                                   if existing_idx < len(planned_order_actual) else None)
            found_pairs.append({
                "new_obj_actual": new_obj_actual,
                "existing_obj_actual": existing_obj_actual,
                "new_obj": normalise(new_obj_actual),
                "existing_obj": normalise(existing_obj_actual),
                "situation": sit,
            })
    return found_pairs


def compute_success(expected_pairs, found_pairs):
    """
    Directional match only: (new_obj, existing_obj) must match exactly.
    sphere->cup and cup->sphere are different relations — no symmetric matching.
    """
    def canon(obj):
        if obj is None:
            return None
        if isinstance(obj, tuple) and len(obj) == 2:
            return obj_to_canonical(*obj)
        return obj

    found_set = set()
    for fp in found_pairs:
        a = canon(fp.get("new_obj"))
        b = canon(fp.get("existing_obj"))
        if a and b:
            found_set.add((a, b))   # directional only

    matched, missing = [], []
    for ep in expected_pairs:
        a = canon(ep.get("new_obj"))
        b = canon(ep.get("existing_obj"))
        if a and b and (a, b) in found_set:
            matched.append(ep)
        else:
            missing.append(ep)

    n_expected = len(expected_pairs)
    n_matched = len(matched)
    pct = 100.0 * n_matched / n_expected if n_expected else 0.0
    return n_expected, n_matched, pct, matched, missing


def get_all_step_rgb_paths(experiment_csv_path, base_image_path=""):
    actual_csv = _find_existing_csv(experiment_csv_path)
    if actual_csv is None:
        return []
    try:
        df = load_csv(actual_csv)
    except Exception:
        return []
    paths = []
    for _, row in df.sort_values("step").iterrows():
        rgb = row.get("rgb_image_path", "")
        if pd.notna(rgb):
            candidates = [str(rgb)]
            if base_image_path:
                candidates.insert(0, os.path.join(base_image_path, str(rgb)))
            for full in candidates:
                if os.path.exists(full):
                    paths.append(full)
                    break
    return paths


def run_simulation_and_check(case_idx, tower_id, steps, cfg, one_shot_to_matched,
                             swap, planned_order_matched, run_label,
                             one_shot_label, expected_pairs_override=None,
                             eval_mode="specific"):
    """
    Run the simulation for a planned order and evaluate the result.

    eval_mode="specific" : check the specific expected pairs were achieved.
    eval_mode="any_pair" : check that ANY task relation was achieved at all.
                           expected_pairs are shown as reference info only.
    """
    task = cfg["eval"]["task"]

    if swap:
        planned_order_actual = list(planned_order_matched)
        planned_order_actual[swap["idx"]] = swap["actual"]
    else:
        planned_order_actual = list(planned_order_matched)

    if expected_pairs_override is not None:
        expected_pairs = expected_pairs_override
    else:
        expected_pairs = get_task_pairs_from_tower(steps, task) if steps else []

    os.makedirs(cfg["_exp_dir"], exist_ok=True)
    get_experiment(planned_order_actual, cfg["_exp_dir"] + "/", f"{task}_{run_label}")
    result_csv = os.path.join(cfg["_exp_dir"], f"{task}_{run_label}.csv")
    result_csv, collapsed = truncate_csv_at_collapse(result_csv)

    if collapsed:
        print("  ✗ COLLAPSE during execution — counted as failed plan case")
        rgb_paths = get_all_step_rgb_paths(result_csv)
        image_path = save_result_image(
            rgb_paths, case_idx, planned_order_actual,
            [], expected_pairs,
            0, 1, 0.0, cfg["_img_dir"], cfg,
            swap=swap, eval_mode=eval_mode)
        print(f"  Image -> {image_path}")
        return {
            "case_idx": case_idx,
            "tower_id": tower_id,
            "eval_mode": eval_mode,
            "one_shot_object": one_shot_label,
            "in_pair_swap": swap["in_pair"] if swap else None,
            "object_order_matched": str(planned_order_matched),
            "object_order_actual": str(planned_order_actual),
            "one_shot_swap": str(swap),
            "plan_status": "SUCCESS",
            "n_expected": 1 if eval_mode == "any_pair" else len(expected_pairs),
            "n_matched": 0,
            "pct": 0.0,
            "status": "COLLAPSE",
            "expected_pairs": "any" if eval_mode == "any_pair" else str(expected_pairs),
            "found_pairs": "[]",
            "missing_pairs": "[]",
            "image_path": image_path,
        }

    found_pairs = check_task_in_result(result_csv, task, planned_order_actual, one_shot_to_matched)

    if eval_mode == "any_pair":
        # Success = at least one task relation found, regardless of which pair
        success = len(found_pairs) > 0
        pct = 100.0 if success else 0.0
        n_expected = 1
        n_matched = 1 if success else 0
        missing_pairs = []
        if success:
            found_str = ", ".join(
                f"{clean_obj_label(fp.get('new_obj_actual', fp.get('new_obj')))}"
                f"->{clean_obj_label(fp.get('existing_obj_actual', fp.get('existing_obj')))}"
                for fp in found_pairs
            )
            print(f"  ✓ ANY_PAIR SUCCESS — found: {found_str}")
        else:
            print(f"  ✗ ANY_PAIR FAIL — no {task} relation found")
    else:
        n_expected, n_matched, pct, matched_list, missing_pairs = compute_success(
            expected_pairs, found_pairs)
        if pct == 100.0:
            print(f"  ✓ FULL SUCCESS — all {n_expected} pair(s) found")
        elif pct > 0:
            print(f"  ~ PARTIAL — {n_matched}/{n_expected} ({pct:.0f}%)")
        else:
            print(f"  ✗ FAIL — 0/{n_expected} pair(s) found")

    rgb_paths = get_all_step_rgb_paths(result_csv)
    image_path = save_result_image(
        rgb_paths, case_idx, planned_order_actual,
        found_pairs, expected_pairs,
        n_matched, n_expected, pct, cfg["_img_dir"], cfg,
        swap=swap, eval_mode=eval_mode)
    print(f"  Image -> {image_path}")

    return {
        "case_idx": case_idx,
        "tower_id": tower_id,
        "eval_mode": eval_mode,
        "one_shot_object": one_shot_label,
        "in_pair_swap": swap["in_pair"] if swap else None,
        "object_order_matched": str(planned_order_matched),
        "object_order_actual": str(planned_order_actual),
        "one_shot_swap": str(swap),
        "plan_status": "SUCCESS",
        "n_expected": n_expected,
        "n_matched": n_matched,
        "pct": round(pct, 1),
        "status": "SUCCESS" if pct == 100.0 else "PARTIAL" if pct > 0 else "FAIL",
        "expected_pairs": "any" if eval_mode == "any_pair" else str(expected_pairs),
        "found_pairs": str(found_pairs),
        "missing_pairs": str(missing_pairs),
        "image_path": image_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_result_image(rgb_paths, case_idx, object_list_actual, found_pairs,
                      expected_pairs, n_matched, n_expected, pct, out_dir, cfg,
                      swap=None, is_no_plan=False, eval_mode="specific"):
    os.makedirs(out_dir, exist_ok=True)
    task = cfg["eval"]["task"]

    target_h, target_w = 240, 320
    step_imgs = []
    if rgb_paths:
        for p in rgb_paths:
            try:
                img = Image.open(p).convert("RGB").resize((target_w, target_h), Image.LANCZOS)
                step_imgs.append(img)
            except Exception:
                pass
    if not step_imgs:
        step_imgs.append(Image.new("RGB", (target_w, target_h), (40, 40, 40)))

    caption_h = 130
    total_w = len(step_imgs) * target_w
    canvas = Image.new("RGB", (total_w, target_h + caption_h), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)

    for i, img in enumerate(step_imgs):
        canvas.paste(img, (i * target_w, 0))
        draw.text((i * target_w + 5, 5), f"Step {i+1}", fill=(255, 255, 255))

    try:
        font   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = font_s = ImageFont.load_default()

    if is_no_plan:
        status_color = (255, 165, 0)
        status_str = "PLANNER_MISS — feasibility order shown"
    elif pct == 100.0:
        status_color = (80, 200, 120)
        status_str = "SUCCESS (100%)"
    elif pct > 0:
        status_color = (230, 160, 50)
        status_str = f"PARTIAL ({pct:.0f}%  {n_matched}/{n_expected})"
    else:
        status_color = (220, 80, 60)
        if eval_mode == "any_pair":
            status_str = f"FAIL (no {task} relation found)"
        else:
            status_str = f"FAIL (0%  0/{n_expected})"

    y = target_h + 6
    draw.text((10, y), f"RESULT: {status_str}  [case {case_idx}]",
              fill=status_color, font=font)
    y += 20

    # Goal line — differs by mode
    if eval_mode == "any_pair":
        draw.text((10, y), f"Goal: any {task} pair (unspecified)",
                  fill=(180, 180, 100), font=font_s)
    else:
        exp_strs = [
            f"{clean_obj_label(ep.get('new_obj'))}->{clean_obj_label(ep.get('existing_obj'))}"
            for ep in expected_pairs
        ]
        draw.text((10, y), f"Expected ({task}): {', '.join(exp_strs)[:130]}",
                  fill=(200, 200, 200), font=font_s)
    y += 16

    found_strs = [
        f"{clean_obj_label(fp.get('new_obj_actual', fp.get('new_obj')))}"
        f"->{clean_obj_label(fp.get('existing_obj_actual', fp.get('existing_obj')))}"
        for fp in found_pairs
    ] if found_pairs else ["none"]
    draw.text((10, y), f"Found ({task}): {', '.join(found_strs)[:130]}",
              fill=status_color, font=font_s)
    y += 16

    obj_str = " > ".join(clean_obj_label(o) for o in object_list_actual)
    draw.text((10, y), obj_str[:130], fill=(160, 160, 160), font=font_s)

    if swap:
        y += 16
        draw.text((10, y),
                  f"ONE-SHOT [pos {swap['idx']}]: "
                  f"{clean_obj_label(swap['matched'])} -> {clean_obj_label(swap['actual'])}",
                  fill=(180, 130, 255), font=font_s)

    file_status = "NOPLAN" if is_no_plan else f"{int(pct):03d}pct"
    safe = re.sub(r"[^\w\-]", "_",
                  f"case_{case_idx:04d}_{eval_mode}_{file_status}_{task}")
    fpath = os.path.join(out_dir, f"{safe}.png")
    canvas.save(fpath)
    return fpath


# ─────────────────────────────────────────────────────────────────────────────
# One-shot helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_match_cache(cfg):
    path = cfg["one_shot"].get("match_cache_path", "match_results_cache.json")
    if not os.path.exists(path):
        if cfg["one_shot"].get("enabled", False):
            sys.exit(f"CRITICAL: one_shot.enabled=true but cache not found: {path}")
        return {}
    with open(path, "r") as f:
        cache = json.load(f)
    print(f"Loaded match cache <- {path} ({len(cache)} entries)")
    return cache


def build_one_shot_maps(match_cache, one_shot_objects):
    matched_to_actual = {}
    one_shot_to_matched = {}
    for obj in one_shot_objects:
        one_shot_canon = obj_to_canonical(*obj)
        result = match_cache.get(str(one_shot_canon), {})
        if result.get("is_novel", True):
            print(f"  [NOVEL] {one_shot_canon} — no reliable match, skipping")
            continue
        if result.get("status") == "already_exists":
            continue
        try:
            matched_canon = tuple(ast.literal_eval(result.get("best_match", "")))
        except Exception:
            print(f"  [WARN] Cannot parse best_match for {one_shot_canon}")
            continue
        matched_to_actual[matched_canon] = obj
        one_shot_to_matched[one_shot_canon] = matched_canon
        print(f"  ONE-SHOT map: {one_shot_canon} -> matched={matched_canon}")
    return matched_to_actual, one_shot_to_matched


def substitute_specific_one_shot(planned_order_matched, one_shot_obj,
                                  matched_canon, expected_pairs, force_in_pair):
    pair_canons = set()
    for ep in expected_pairs:
        for k in ("new_obj", "existing_obj"):
            obj = ep.get(k)
            if obj:
                pair_canons.add(obj_to_canonical(*obj))

    eligible = [i for i, obj in enumerate(planned_order_matched)
                if obj_to_canonical(*obj) == matched_canon]
    if not eligible:
        return list(planned_order_matched), None

    in_pair_positions = [i for i in eligible if matched_canon in pair_canons]
    target_idx = random.choice(in_pair_positions if force_in_pair and in_pair_positions else eligible)

    result = list(planned_order_matched)
    obj_to_replace = result[target_idx]
    result[target_idx] = one_shot_obj

    swap = {
        "matched": obj_to_replace,
        "actual": one_shot_obj,
        "matched_canon": matched_canon,
        "idx": target_idx,
        "in_pair": matched_canon in pair_canons,
    }
    print(f"  [ONE-SHOT SWAP] pos {target_idx}: {matched_canon} -> "
          f"{obj_to_canonical(*one_shot_obj)} (in_pair={swap['in_pair']})")
    return result, swap


# ─────────────────────────────────────────────────────────────────────────────
# Tuple-limit / tuple-pair filtering
# ─────────────────────────────────────────────────────────────────────────────

def tuple_key(obj):
    obj_type, size_or_path = obj
    try:
        val = float(size_or_path)
        return (str(obj_type), str(val))
    except Exception:
        return (str(obj_type), os.path.basename(str(size_or_path)))


def load_tuple_constraints(cfg):
    tc = cfg.get("tuple_constraints", {})
    if tc.get("enabled", True) is False:
        return {}, []
    limits = dict(DEFAULT_TUPLE_LIMITS)
    pairs = list(DEFAULT_TUPLE_PAIRS)
    if "limits" in tc:
        limits = {}
        for item in tc.get("limits", []):
            obj_type, name, limit = item
            limits[(str(obj_type), str(name))] = int(limit)
    if "pairs" in tc:
        pairs = []
        for a, b in tc.get("pairs", []):
            pairs.append((tuple(a), tuple(b)))
    return limits, pairs


def obeys_tuple_constraints(object_list, tuple_limits, tuple_pairs):
    if not tuple_limits and not tuple_pairs:
        return True
    pair_group = {}
    if tuple_pairs:
        for a, b in tuple_pairs:
            a = tuple(a)
            b = tuple(b)
            group_key = tuple(sorted([a, b]))
            pair_group[a] = group_key
            pair_group[b] = group_key
    counts = defaultdict(int)
    for obj in object_list:
        canon = tuple_key(obj)
        key = pair_group.get(canon, canon)
        counts[key] += 1
        if canon in pair_group:
            pair = pair_group[canon]
            shared_limit = min(tuple_limits.get(tuple(x), float("inf")) for x in pair)
            if counts[key] > shared_limit:
                return False
        else:
            limit = tuple_limits.get(canon, float("inf"))
            if counts[key] > limit:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _report_mode(rows, mode_label):
    if not rows:
        print(f"      (no {mode_label} cases)")
        return
    df = pd.DataFrame(rows)
    status_col = df.get("status", pd.Series(dtype=str))
    planner_miss = int((status_col == "PLANNER_MISS").sum())
    n_full      = int((status_col == "SUCCESS").sum())
    n_partial   = int((status_col == "PARTIAL").sum())
    n_fail      = int((status_col == "FAIL").sum())
    n_collapse  = int((status_col == "COLLAPSE").sum())
    runnable    = df[~status_col.isin(["PLANNER_MISS"])]
    runnable_pct = runnable["pct"].mean() if len(runnable) and "pct" in runnable.columns else 0.0
    print(f"      [{mode_label}]  total={len(df)}  "
          f"success={n_full}  partial={n_partial}  fail={n_fail}  "
          f"collapse={n_collapse}  planner_miss={planner_miss}  "
          f"avg_pct(ran)={runnable_pct:.1f}%")


def print_report(label, rows):
    print(f"\n  -- {label} --")
    if not rows:
        print("    (no cases)")
        return
    df = pd.DataFrame(rows)
    total = len(df)
    status_col = df.get("status", pd.Series(dtype=str))
    planner_miss = int((status_col == "PLANNER_MISS").sum())
    n_full      = int((status_col == "SUCCESS").sum())
    n_partial   = int((status_col == "PARTIAL").sum())
    n_fail      = int((status_col == "FAIL").sum())
    n_collapse  = int((status_col == "COLLAPSE").sum())
    avg_pct     = df["pct"].mean() if "pct" in df.columns else 0.0
    runnable    = df[~status_col.isin(["PLANNER_MISS"])] if "status" in df.columns else df
    runnable_pct = runnable["pct"].mean() if len(runnable) and "pct" in runnable.columns else 0.0

    print(f"    Total rows            : {total}")
    print(f"    Planner miss (valid)  : {planner_miss}  <- solvable problem, planner failed")
    print(f"    Collapse during exec  : {n_collapse}")
    print(f"    Success  (100%)       : {n_full}")
    print(f"    Partial  (>0%)        : {n_partial}")
    print(f"    Fail     (0%)         : {n_fail}")
    print(f"    Avg pct (all)         : {avg_pct:.1f}%")
    print(f"    Avg pct (ran)         : {runnable_pct:.1f}%")

    if "eval_mode" in df.columns and df["eval_mode"].nunique() > 1:
        print(f"    --- by eval mode ---")
        for mode in sorted(df["eval_mode"].dropna().unique()):
            _report_mode(rows=[r for r in rows if r.get("eval_mode") == mode], mode_label=mode)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/eval.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = cfg["eval"]["task"]
    seed = int(cfg["eval"].get("seed", 42))
    set_seed(seed)

    os.makedirs(cfg["_pddl_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["_pddl_dir"], "successful_plans"), exist_ok=True)
    os.makedirs(os.path.join(cfg["_pddl_dir"], "no_plan_files"), exist_ok=True)
    os.makedirs(cfg["_eval_dir"], exist_ok=True)
    os.makedirs(cfg["_exp_dir"], exist_ok=True)
    os.makedirs(cfg["_img_dir"], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  model  : {cfg['model_name']}")
    print(f"  task   : {task}")
    print(f"  domain : {cfg['_domain']}")
    print(f"  output : {cfg['_eval_dir']}")
    print(f"{'='*60}\n")

    if not os.path.exists(cfg["_domain"]):
        sys.exit(f"CRITICAL: domain.pddl not found at {cfg['_domain']}\n"
                 f"Run learn_rules first for model '{cfg['model_name']}'.")

    # 1. Load model
    print(f"--> Loading model: {cfg['model_name']}")
    model, train_cfg = load_ckpt(cfg["model_name"], tag="best")
    cfg["_train_cfg"] = train_cfg
    _load_data.init(train_cfg)
    project_set_seed(train_cfg["seed"])
    model.freeze(deterministic=True)

    rel_dim = train_cfg["model"]["symbol_size"]
    obj_dim = train_cfg["model"]["obj_symbol_size"]

    # 2. Symbols
    print("--> Computing collapse symbols...")
    collapse_syms = get_collapse_symbols(model, cfg)
    print(f"  collapse_symbols: {collapse_syms}")

    print(f"--> Computing task relation symbols for '{task}'...")
    task_pos_rel_symbols = get_task_relation_symbols(model, cfg, task)
    if not task_pos_rel_symbols:
        sys.exit(f"CRITICAL: No positive relation symbols found for task={task}.")

    # 3. Dataset / sampling
    dataset_csv = cfg["data"]["dataset_csv"]
    print(f"\n--> Loading data: {dataset_csv}")
    df = load_csv(dataset_csv)
    print(f"  {len(df)} rows loaded")

    print(f"--> Finding towers with at least one '{task}' situation...")
    qualifying_all = find_qualifying_towers(df, task)
    print(f"  {len(qualifying_all)} towers contain task='{task}' before filters")

    tuple_limits, tuple_pairs = load_tuple_constraints(cfg)
    if tuple_limits or tuple_pairs:
        before = len(qualifying_all)
        qualifying_all = {
            tid: steps
            for tid, steps in qualifying_all.items()
            if obeys_tuple_constraints(build_object_list_from_steps(steps), tuple_limits, tuple_pairs)
        }
        print(f"  {len(qualifying_all)} towers after tuple-limit filtering  "
              f"(removed {before - len(qualifying_all)})")

    min_objects = cfg["eval"].get("min_objects")
    max_objects = cfg["eval"].get("max_objects")
    min_objects = int(min_objects) if min_objects is not None else None
    max_objects = int(max_objects) if max_objects is not None else None

    n_baseline = int(cfg["eval"].get("n_baseline_scenarios", 5))
    feasibility_tries = int(cfg["eval"].get("feasibility_tries", 30))

    # synthetic_mode: build cases for every N in [min_objects..max_objects].
    # Activates whenever max_objects is set.
    synthetic_mode = max_objects is not None

    if synthetic_mode:
        min_n = min_objects if min_objects is not None else 2
        max_n = max_objects

        print(f"  Synthetic mode: building object groups for N={min_n}..{max_n}")
        base_fragments = build_base_fragments(qualifying_all, task, tuple_limits, tuple_pairs)
        print(f"  base fragments with at least one task pair: {len(base_fragments)}")
        print(f"  base fragment length distribution: "
              f"{Counter(len(f['objects']) for f in base_fragments)}")

        baseline_cases = []
        for target_n in range(min_n, max_n + 1):
            eligible_fragments = [f for f in base_fragments if len(f["objects"]) <= target_n]
            if not eligible_fragments:
                print(f"  [SKIP N={target_n}] No base fragments with <={target_n} objects — "
                      f"smallest has {min(len(f['objects']) for f in base_fragments)}")
                continue
            cases_n = build_synthetic_cases(
                base_fragments, target_n, n_baseline, tuple_limits, tuple_pairs,
                max_attempts_per_case=int(cfg["eval"].get("synthetic_max_attempts", 2000)),
            )
            for c in cases_n:
                c["target_n"] = target_n
                c["case_id"] = f"N{target_n}_{c['case_id']}"
            baseline_cases.extend(cases_n)
            print(f"  synthetic cases built for N={target_n}: {len(cases_n)}/{n_baseline}")

        qualifying = qualifying_all
        task_groups = {}
    else:
        qualifying = qualifying_all
        if min_objects is not None or max_objects is not None:
            qualifying = {
                tid: steps
                for tid, steps in qualifying.items()
                if (
                    (min_objects is None or len(build_object_list_from_steps(steps)) >= min_objects)
                    and
                    (max_objects is None or len(build_object_list_from_steps(steps)) <= max_objects)
                )
            }
        print(f"  {len(qualifying)} qualifying towers after object-count filtering")
        if not qualifying:
            print("No qualifying towers. Exiting.")
            return
        real_cases = make_real_cases(qualifying, task)
        print("--> Building task groups for diverse sampling...")
        grouped_keys = {case["group_key"] for case in real_cases}
        print(f"  {len(grouped_keys)} unique (object_pool, pair) groups")
        baseline_cases = sample_diverse_cases(real_cases, n_baseline)
        task_groups = build_task_groups(qualifying, task)

    if not baseline_cases:
        print("No planning cases could be built. Exiting.")
        return

    # 4. Object symbols
    print("\n--> Pre-computing object symbols once...")
    all_objects = []
    for case in baseline_cases:
        all_objects.extend(case["object_list"])
    if cfg["one_shot"].get("enabled", False):
        all_objects.extend([tuple(o) for o in cfg["one_shot"].get("objects", DEFAULT_ONE_SHOT_EXTRA)])
    object_symbol_cache = build_object_symbol_cache(all_objects, model, cfg)
    print(f"  object symbol cache: {len(object_symbol_cache)} entries")

    # 5. One-shot maps
    one_shot_enabled = bool(cfg["one_shot"].get("enabled", False))
    if one_shot_enabled:
        print("\n--> Loading one-shot match cache...")
        match_cache = load_match_cache(cfg)
        one_shot_extra = [tuple(o) for o in cfg["one_shot"].get("objects", DEFAULT_ONE_SHOT_EXTRA)]
        matched_to_actual, one_shot_to_matched = build_one_shot_maps(match_cache, one_shot_extra)
        print(f"  {len(matched_to_actual)} swappable one-shot object(s)")
    else:
        matched_to_actual = {}
        one_shot_to_matched = {}
        print("\n--> One-shot disabled")

    results = []
    case_idx = 0

    # =========================================================================
    # PHASE 1: baseline
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 1: Baseline — {len(baseline_cases)} scenarios")
    print(f"{'='*60}")

    baseline_idx = 0
    while baseline_idx < len(baseline_cases):
        case = baseline_cases[baseline_idx]
        baseline_idx += 1

        tower_id = case["case_id"]
        steps = case.get("steps")
        object_list_matched = case["object_list"]
        expected_pairs = case["expected_pairs"]

        print(f"\n[baseline {case_idx}] case_id={tower_id} source={case.get('source')}")
        if case.get("sources"):
            print(f"  Synthetic fragments: {case['sources']}")
        print(f"  Objects ({len(object_list_matched)}): {object_list_matched}")
        print(f"  Expected task pairs: {len(expected_pairs)}")

        missing = [o for o in object_list_matched
                   if object_symbol_cache.get(obj_to_canonical(*o)) is None]
        if missing:
            print(f"  [SKIP] Missing symbols for: {missing}")
            case_idx += 1
            continue

        obj_symbols = [object_symbol_cache[obj_to_canonical(*o)] for o in object_list_matched]
        req_sym_list = _make_required_goal_cases(
            expected_pairs, task_pos_rel_symbols, object_symbol_cache)

        if not req_sym_list:
            print("  [SKIP] No valid required-symbol goals for this case")
            case_idx += 1
            continue

        run_label = f"baseline_{case_idx:04d}_{tower_id}"

        # ── Helper: handle NO_PLAN for either mode ────────────────────────
        def _handle_no_plan(plan_status, full_output, mode_label):
            """
            Runs feasibility check on NO_PLAN.
            Returns (should_discard, planner_miss_recorded).
            """
            print(f"  [{mode_label}] Planner returned {plan_status} — "
                  f"running feasibility check ({feasibility_tries} tries)...")

            achievable, best_order = is_achievable_order(
                object_list_matched, expected_pairs, task, cfg,
                n_tries=feasibility_tries,
            )

            if not achievable:
                print(f"  [{mode_label}] [INVALID CASE] No ordering achieves the task.")
                return True, False

            print(f"  [{mode_label}] [PLANNER_MISS] A valid ordering exists — planner failed.")
            print(f"  Best order: {' > '.join(clean_obj_label(o) for o in best_order)}")

            diag_label = f"PLANNERMISS_{mode_label}_{run_label}"
            get_experiment(best_order, cfg["_exp_dir"] + "/", diag_label)
            diag_csv = os.path.join(cfg["_exp_dir"], f"{diag_label}.csv")
            diag_csv, _ = truncate_csv_at_collapse(diag_csv)

            diag_found_pairs = check_task_in_result(
                diag_csv, task, best_order, one_shot_to_matched)
            n_exp, n_mat, pct_, ml, miss = compute_success(expected_pairs, diag_found_pairs)
            diag_rgb = get_all_step_rgb_paths(diag_csv)
            img_path = save_result_image(
                diag_rgb, case_idx, best_order, diag_found_pairs, expected_pairs,
                n_mat, n_exp, 0.0, cfg["_img_dir"], cfg,
                swap=None, is_no_plan=True, eval_mode=mode_label)

            results.append({
                "case_idx": case_idx,
                "tower_id": tower_id,
                "eval_mode": mode_label,
                "case_source": case.get("source"),
                "one_shot_object": "baseline",
                "in_pair_swap": None,
                "object_order_matched": str(object_list_matched),
                "object_order_actual": str(best_order),
                "one_shot_swap": str(None),
                "plan_status": plan_status,
                "n_expected": n_exp,
                "n_matched": n_mat,
                "pct": round(pct_, 1),
                "status": "PLANNER_MISS",
                "expected_pairs": str(expected_pairs),
                "found_pairs": str(diag_found_pairs),
                "missing_pairs": str(miss),
                "image_path": img_path,
                "planner_output": full_output[:600],
            })
            return False, True

        # ── MODE A: specific pair goal ────────────────────────────────────
        print("\n  [MODE A] specific pair goal")
        try:
            problem_path_a = construct_task_problem_pddl(
                obj_symbols, collapse_syms, req_sym_list,
                rel_dim, obj_dim, cfg["_pddl_dir"], run_label)
        except Exception as e:
            print(f"  [ERROR] PDDL gen (mode A): {e}")
            results.append({
                "case_idx": case_idx, "tower_id": tower_id, "eval_mode": "specific",
                "one_shot_object": "baseline", "in_pair_swap": None,
                "status": f"PDDL_ERROR:{e}", "pct": 0.0,
            })
            case_idx += 1
            continue

        plan_status_a, plan_lines_a, output_a = run_planner(problem_path_a, cfg)
        print(f"  [MODE A] Planner: {plan_status_a}")
        save_planner_artifacts(problem_path_a, plan_status_a, cfg, f"A_{run_label}")

        if plan_status_a != "SUCCESS":
            discard, _ = _handle_no_plan(plan_status_a, output_a, "specific")
            if discard:
                # Object group invalid — try to get a replacement
                if synthetic_mode:
                    replacement = None
                    _case_target_n = case.get("target_n", len(object_list_matched))
                    for _rep_try in range(10):
                        _cand = build_synthetic_case(
                            len(baseline_cases), base_fragments, _case_target_n,
                            tuple_limits, tuple_pairs,
                            max_attempts=int(cfg["eval"].get("synthetic_max_attempts", 2000)),
                        )
                        if _cand is None:
                            break
                        _cand["target_n"] = _case_target_n
                        _cand["case_id"] = f"N{_case_target_n}_{_cand['case_id']}"
                        _ok, _ord = is_achievable_order(
                            _cand["object_list"], _cand["expected_pairs"],
                            task, cfg, n_tries=feasibility_tries)
                        if _ok:
                            _cand["_feasibility_best_order"] = _ord
                            replacement = _cand
                            break
                        print("  [WARN] replacement candidate not achievable, retrying...")
                    if replacement is not None:
                        baseline_cases.append(replacement)
                        print(f"  [REPLACED] Added {replacement['case_id']} to baseline queue")
                    else:
                        print("  [WARN] Could not generate a valid replacement case.")
                continue   # do NOT increment case_idx
        else:
            planned_order_a = extract_order_from_plan(plan_lines_a, object_list_matched)
            print(f"  [MODE A] Plan: {' > '.join(clean_obj_label(o) for o in planned_order_a)}")
            row_a = run_simulation_and_check(
                case_idx, tower_id, steps, cfg, one_shot_to_matched,
                swap=None, planned_order_matched=planned_order_a,
                run_label=f"A_{run_label}", one_shot_label="baseline",
                expected_pairs_override=expected_pairs,
                eval_mode="specific")
            row_a["planner_output"] = output_a[:600]
            row_a["case_source"] = case.get("source")
            if case.get("sources"):
                row_a["synthetic_sources"] = str(case.get("sources"))
            results.append(row_a)

        # ── MODE B: any-pair goal ─────────────────────────────────────────
        n_pairs = len(obj_symbols) * (len(obj_symbols) - 1)
        print(f"\n  [MODE B] any-pair goal  ({n_pairs} ordered pairs)")
        try:
            problem_path_b = construct_task_problem_pddl_any(
                obj_symbols, collapse_syms, task_pos_rel_symbols,
                rel_dim, obj_dim, cfg["_pddl_dir"], run_label)
        except Exception as e:
            print(f"  [ERROR] PDDL gen (mode B): {e}")
            results.append({
                "case_idx": case_idx, "tower_id": tower_id, "eval_mode": "any_pair",
                "one_shot_object": "baseline", "in_pair_swap": None,
                "status": f"PDDL_ERROR:{e}", "pct": 0.0,
            })
            case_idx += 1
            continue

        plan_status_b, plan_lines_b, output_b = run_planner(problem_path_b, cfg)
        print(f"  [MODE B] Planner: {plan_status_b}")
        save_planner_artifacts(problem_path_b, plan_status_b, cfg, f"B_{run_label}")

        if plan_status_b != "SUCCESS":
            _handle_no_plan(plan_status_b, output_b, "any_pair")
        else:
            planned_order_b = extract_order_from_plan(plan_lines_b, object_list_matched)
            print(f"  [MODE B] Plan: {' > '.join(clean_obj_label(o) for o in planned_order_b)}")
            row_b = run_simulation_and_check(
                case_idx, tower_id, steps, cfg, one_shot_to_matched,
                swap=None, planned_order_matched=planned_order_b,
                run_label=f"B_{run_label}", one_shot_label="baseline",
                expected_pairs_override=None,   # any_pair: no specific target
                eval_mode="any_pair")
            row_b["planner_output"] = output_b[:600]
            row_b["case_source"] = case.get("source")
            if case.get("sources"):
                row_b["synthetic_sources"] = str(case.get("sources"))
            results.append(row_b)

        case_idx += 1

    # =========================================================================
    # PHASE 2: one-shot variants
    # =========================================================================
    if one_shot_enabled and synthetic_mode:
        print("\n[WARN] one-shot evaluation is skipped in synthetic mode.")

    if one_shot_enabled and not synthetic_mode:
        n_each = int(cfg["one_shot"].get("n_scenarios_per_one_shot", 20))
        in_pair_ratio = float(cfg["one_shot"].get("in_pair_ratio", 0.0))
        one_shot_extra = [tuple(o) for o in cfg["one_shot"].get("objects", DEFAULT_ONE_SHOT_EXTRA)]

        for one_shot_obj in one_shot_extra:
            one_shot_canon = obj_to_canonical(*one_shot_obj)
            matched_canon = one_shot_to_matched.get(one_shot_canon)
            if matched_canon is None:
                print(f"\n[SKIP one-shot] {one_shot_canon} — novel or no match")
                continue

            one_shot_label = str(one_shot_canon)
            n_in_pair = math.ceil(in_pair_ratio * n_each)
            n_other = n_each - n_in_pair

            print(f"\n{'='*60}")
            print(f"PHASE 2: one-shot={one_shot_label} matched={matched_canon}")
            print(f"  {n_in_pair} in-pair cases + {n_other} other cases")
            print(f"{'='*60}")

            pair_ids, nonpair_ids = get_pair_tower_ids(qualifying, matched_canon, task)
            pair_groups = filter_task_groups(task_groups, pair_ids)
            nonpair_groups = filter_task_groups(task_groups, nonpair_ids)
            in_pair_towers = sample_diverse_towers(pair_groups, n_in_pair)
            other_towers = sample_diverse_towers(nonpair_groups, n_other)
            tower_schedule = ([(tid, True) for tid in in_pair_towers] +
                              [(tid, False) for tid in other_towers])
            random.shuffle(tower_schedule)
            actual_in_pair = 0

            for tower_id, force_in_pair in tower_schedule:
                steps = qualifying[tower_id]
                print(f"\n[{one_shot_label} case {case_idx}] tower_id={tower_id} "
                      f"force_in_pair={force_in_pair}")

                object_list_matched = build_object_list_from_steps(steps)
                expected_pairs = get_task_pairs_from_tower(steps, task)
                missing = [o for o in object_list_matched
                           if object_symbol_cache.get(obj_to_canonical(*o)) is None]
                if missing:
                    print(f"  [SKIP] Missing symbols for: {missing}")
                    continue

                obj_symbols = [object_symbol_cache[obj_to_canonical(*o)]
                               for o in object_list_matched]
                req_sym_list = _make_required_goal_cases(
                    expected_pairs, task_pos_rel_symbols, object_symbol_cache)
                if not req_sym_list:
                    print("  [SKIP] No valid required-symbol goals for this case")
                    continue

                safe_os = re.sub(r"[^\w]", "_", one_shot_label)
                run_label = f"oneshot_{safe_os}_{case_idx:04d}_tower_{tower_id}"

                try:
                    problem_path = construct_task_problem_pddl(
                        obj_symbols, collapse_syms, req_sym_list,
                        rel_dim, obj_dim, cfg["_pddl_dir"], run_label)
                except Exception as e:
                    print(f"  [ERROR] PDDL gen: {e}")
                    results.append({
                        "case_idx": case_idx, "tower_id": tower_id,
                        "one_shot_object": one_shot_label, "in_pair_swap": None,
                        "status": f"PDDL_ERROR:{e}", "pct": 0.0,
                    })
                    case_idx += 1
                    continue

                plan_status, plan_lines, full_output = run_planner(problem_path, cfg)
                print(f"  Planner: {plan_status}")
                save_planner_artifacts(problem_path, plan_status, cfg, run_label)

                if plan_status != "SUCCESS":
                    print(f"  Planner returned {plan_status} — running feasibility check "
                          f"({feasibility_tries} tries)...")
                    achievable, best_order = is_achievable_order(
                        object_list_matched, expected_pairs, task, cfg,
                        n_tries=feasibility_tries)
                    if not achievable:
                        print("  [INVALID CASE] No ordering achieves the task — skipping.")
                        continue
                    print(f"  [PLANNER_MISS] A valid ordering exists — planner/domain failed.")
                    print(f"  Best order: {' > '.join(clean_obj_label(o) for o in best_order)}")
                    diag_label = f"PLANNERMISS_{run_label}"
                    get_experiment(best_order, cfg["_exp_dir"] + "/", diag_label)
                    diag_csv = os.path.join(cfg["_exp_dir"], f"{diag_label}.csv")
                    diag_csv, _ = truncate_csv_at_collapse(diag_csv)
                    diag_found_pairs = check_task_in_result(
                        diag_csv, task, best_order, one_shot_to_matched)
                    n_expected, n_matched, pct, matched_list, missing_pairs = compute_success(
                        expected_pairs, diag_found_pairs)
                    diag_rgb = get_all_step_rgb_paths(diag_csv)
                    image_path = save_result_image(
                        diag_rgb, case_idx, best_order, diag_found_pairs, expected_pairs,
                        n_matched, n_expected, 0.0, cfg["_img_dir"], cfg,
                        swap=None, is_no_plan=True, eval_mode="specific")
                    results.append({
                        "case_idx": case_idx, "tower_id": tower_id,
                        "eval_mode": "specific",
                        "one_shot_object": one_shot_label, "in_pair_swap": None,
                        "object_order_matched": str(object_list_matched),
                        "object_order_actual": str(best_order),
                        "one_shot_swap": str(None), "plan_status": plan_status,
                        "n_expected": n_expected, "n_matched": n_matched,
                        "pct": round(pct, 1), "status": "PLANNER_MISS",
                        "expected_pairs": str(expected_pairs),
                        "found_pairs": str(diag_found_pairs),
                        "missing_pairs": str(missing_pairs),
                        "image_path": image_path,
                        "planner_output": full_output[:600],
                    })
                    case_idx += 1
                    continue

                planned_order_matched = extract_order_from_plan(plan_lines, object_list_matched)
                print(f"  Plan: {' > '.join(clean_obj_label(o) for o in planned_order_matched)}")

                planned_order_actual, swap = substitute_specific_one_shot(
                    planned_order_matched, one_shot_obj, matched_canon,
                    expected_pairs, force_in_pair)
                if swap is None:
                    print(f"  [SKIP] matched_canon {matched_canon} not in plan")
                    continue
                if swap["in_pair"]:
                    actual_in_pair += 1

                row = run_simulation_and_check(
                    case_idx, tower_id, steps, cfg, one_shot_to_matched,
                    swap=swap, planned_order_matched=planned_order_matched,
                    run_label=run_label, one_shot_label=one_shot_label,
                    eval_mode="specific")
                row["planner_output"] = full_output[:600]
                results.append(row)
                case_idx += 1

            ran = len([r for r in results
                       if r.get("one_shot_object") == one_shot_label
                       and r.get("plan_status") == "SUCCESS"])
            if ran:
                print(f"\n  [{one_shot_label}] in-pair swaps: {actual_in_pair}/{ran} "
                      f"({100 * actual_in_pair / ran:.0f}%)")

    # ── Save + report ─────────────────────────────────────────────────────────
    df_r = pd.DataFrame(results)
    df_r.to_csv(cfg["_results"], index=False)
    print(f"\n  CSV -> {cfg['_results']}")

    print(f"\n{'='*60}")
    print(f"EVALUATION REPORTS (TASK={task})")
    print(f"{'='*60}")

    print_report("TOTAL (all cases)", results)
    baseline_rows = [r for r in results if r.get("one_shot_object") == "baseline"]
    print_report("BASELINE (no one-shot swap)", baseline_rows)
    oneshot_rows = [r for r in results if r.get("one_shot_object") != "baseline"]
    print_report("ALL ONE-SHOT (combined)", oneshot_rows)
    for label in sorted({r.get("one_shot_object") for r in oneshot_rows}):
        print_report(f"ONE-SHOT: {label}",
                     [r for r in oneshot_rows if r.get("one_shot_object") == label])

    print(f"\n  Images -> {cfg['_img_dir']}")


if __name__ == "__main__":
    main()