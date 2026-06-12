"""
plan_height_fast.py — shortest / tallest tower evaluation.
Usage:  python plan_height_fast.py -c configs/eval_height.yaml
"""
import argparse, ast, os, random, sys, time
from collections import defaultdict, Counter

import pandas as pd, yaml

from eval_utils import (canonical, clean, load_csv, load_model, get_obj_symbols,
                        get_collapse_syms, run_planner, plan_order, run_sim,
                        parse_bbox_df, obj_init_pddl, rel_init_pddl,
                        save_image_height, sample_diverse, print_summary, row_collapsed)
from load_data import set_seed

# ── config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f: cfg = yaml.safe_load(f)
    root = os.path.join("save", cfg["model_name"], "eval_height")
    cfg.update(_domain   =os.path.join("save", cfg["model_name"], "domain.pddl"),
               _pddl_dir =os.path.join(root, "pddl"),
               _plan_file=os.path.join(root, "pddl", "sas_plan"),
               _exp_dir  =os.path.join(root, "sim"),
               _img_dir  =os.path.join(root, "images"),
               _results  =os.path.join(root, "results.csv"))
    return cfg

# ── height computation ────────────────────────────────────────────────────────

def compute_levels(step_bboxes, threshold):
    levels = 0
    for step in sorted(step_bboxes):
        bbox    = step_bboxes[step]
        new_idx = step - 1 if (step - 1) in bbox else max(bbox.keys())
        new_z   = bbox[new_idx]["max"][2]
        if levels == 0:
            levels = 1
        else:
            prev = [bbox[i]["max"][2] for i in bbox if i != new_idx]
            if prev and new_z > max(prev) + threshold: levels += 1
    return levels

def tower_levels(rows, z_thr):
    """Return height levels for a tower's rows, or None if collapsed."""
    step_bboxes = {}
    for row in rows:
        if row_collapsed(row): return None
        step = int(row.get("step", 0))
        raw  = row.get("bbox")
        if isinstance(raw, float) and pd.isna(raw): continue
        try:
            bbox = ast.literal_eval(str(raw)) if not isinstance(raw, dict) else raw
            if bbox: step_bboxes[step] = {int(k): v for k, v in bbox.items()}
        except Exception: pass
    return compute_levels(step_bboxes, z_thr) if step_bboxes else None

# ── build cases ───────────────────────────────────────────────────────────────

def build_cases(df, min_n, max_n, z_thr):
    by_tower = defaultdict(list)
    for _, row in df.iterrows(): by_tower[row["id"]].append(row.to_dict())
    for rows in by_tower.values(): rows.sort(key=lambda r: int(r["step"]))

    verify_index = defaultdict(dict)
    for rows in by_tower.values():
        for n in range(min_n, max_n + 1):
            if len(rows) < n: continue
            prefix = rows[:n]
            lvl    = tower_levels(prefix, z_thr)
            if lvl is None: continue
            objs = tuple(canonical(str(r["object_type"]), str(r["object_size_or_path"])) for r in prefix)
            verify_index[(tuple(sorted(objs)), n)][objs] = lvl

    cases = []
    for n in range(min_n, max_n + 1):
        by_pool = defaultdict(list)
        for rows in by_tower.values():
            if len(rows) < n: continue
            prefix = rows[:n]
            lvl    = tower_levels(prefix, z_thr)
            if lvl is None: continue
            objs = [(str(r["object_type"]), str(r["object_size_or_path"])) for r in prefix]
            by_pool[tuple(sorted(canonical(*o) for o in objs))].append((rows[0]["id"], objs, lvl))
        for pool, entries in by_pool.items():
            lvls = [e[2] for e in entries]
            min_h, max_h = min(lvls), max(lvls)
            if min_h == max_h: continue
            cases.append(dict(n_objects=n, pool=pool,
                              objs=sorted(entries[0][1], key=lambda o: canonical(*o)),
                              min_h=min_h, max_h=max_h))
    return cases, verify_index

# ── PDDL problem ──────────────────────────────────────────────────────────────

def write_problem(obj_syms, rel_dim, obj_dim, target_h, path):
    n     = len(obj_syms)
    usage = f"(active-count-{n})" if n < 4 else "(all-used)"
    prob  = (f"(define (problem blocks-problem)\n\t(:domain blocks)\n"
             f"\t(:objects\n\t\t{' '.join(f'obj{i}' for i in range(n))} - object\n\t)\n"
             f"\t(:init\n\t\t(H0)\n\t\t(active-count-0)\n"
             f"{obj_init_pddl(obj_syms, obj_dim)}{rel_init_pddl(n, rel_dim)}\t)\n"
             f"\t(:goal (and\n\t\t(H{target_h})\n\t\t{usage}\n"
             f"\t\t(not (active-count-collapse))\n\t))\n)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(prob)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/eval_height.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["eval"].get("seed", 42))
    set_seed(seed); random.seed(seed)
    for d in [cfg["_pddl_dir"], cfg["_exp_dir"], cfg["_img_dir"]]: os.makedirs(d, exist_ok=True)
    if not os.path.exists(cfg["_domain"]): sys.exit(f"domain.pddl not found: {cfg['_domain']}")

    model, train_cfg = load_model(cfg)
    rel_dim = train_cfg["model"]["symbol_size"]
    obj_dim = train_cfg["model"]["obj_symbol_size"]
    print(f"  collapse symbols: {get_collapse_syms(model, train_cfg)}")

    min_n = int(cfg["eval"].get("min_objects", 2))
    max_n = int(cfg["eval"].get("max_objects", 4))
    z_thr = float(cfg["eval"].get("z_level_threshold", 0.15))
    df    = load_csv(cfg["data"]["dataset_csv"])
    print(f"\n→ {len(df)} rows. Building cases N={min_n}..{max_n}...")

    all_cases, verify_index = build_cases(df, min_n, max_n, z_thr)
    per_n = Counter(c["n_objects"] for c in all_cases)
    print(f"  {len(all_cases)} cases with min≠max height  " +
          "  ".join(f"N={n}:{c}" for n, c in sorted(per_n.items())))

    n_cap = int(cfg["eval"].get("n_scenarios_per_n", 0))
    if n_cap > 0:
        all_cases = sample_diverse(all_cases, n_cap)
        print(f"  After sampling: {len(all_cases)} cases")

    unique_objs = list({canonical(*o) for c in all_cases for o in c["objs"]})
    sym_cache   = dict(zip(unique_objs, get_obj_symbols(unique_objs, model, cfg["data"]["depth_image_dir"])))
    print(f"  {len(sym_cache)} unique object symbols cached\n")

    simulate = bool(cfg["eval"].get("simulate", True))
    print(f"── Planning & {'Simulating' if simulate else 'Verifying'} ({len(all_cases)} pools × 2) ──")

    results, t_plan_total, t_sim_total, t_start = [], 0.0, 0.0, time.time()

    for i, case in enumerate(all_cases):
        objs     = case["objs"]
        obj_syms = [sym_cache.get(canonical(*o)) for o in objs]
        if any(s is None for s in obj_syms): print(f"  [{i}] SKIP — missing symbol"); continue

        for target_h, goal_tag in [(case["min_h"], "short"), (case["max_h"], "tall")]:
            label     = f"{i:04d}_N{case['n_objects']}_{goal_tag}_H{target_h}"
            prob_path = os.path.join(cfg["_pddl_dir"], f"p_{label}.pddl")
            write_problem(obj_syms, rel_dim, obj_dim, target_h, prob_path)

            t0 = time.time()
            status, lines = run_planner(prob_path, cfg)
            t_plan = time.time() - t0; t_plan_total += t_plan
            print(f"  [{i}] N={case['n_objects']} H{target_h} ({goal_tag}) {status} ({t_plan:.1f}s)")

            base = dict(pool=str(case["pool"]), n_objects=case["n_objects"], goal=goal_tag,
                        target_h=target_h, min_h=case["min_h"], max_h=case["max_h"],
                        t_plan_sec=round(t_plan, 2))

            if status != "SUCCESS":
                img = save_image_height(cfg["_img_dir"], label, objs, target_h, None, "NO_PLAN", [])
                results.append({**base, "plan_status": status, "plan_order": "",
                                 "sim_h": None, "success": False, "status": "NO_PLAN",
                                 "t_sim_sec": 0.0, "image_path": img}); continue

            ordered   = plan_order(lines, objs)
            order_str = " > ".join(clean(o) for o in ordered)
            print(f"    order: {order_str}")
            base["plan_order"] = order_str

            if not simulate:
                t0 = time.time()
                order_key = tuple(canonical(*o) for o in ordered)
                known     = verify_index.get((case["pool"], case["n_objects"]), {})
                data_lvl  = known.get(order_key)
                if data_lvl is None:
                    v_status = "UNVERIFIED"
                    v_note   = f"order unseen; {sum(1 for l in known.values() if l==target_h)} known seqs hit H{target_h}"
                elif data_lvl == target_h:
                    v_status, v_note = "VERIFIED",  f"exact order in data H={data_lvl}"
                else:
                    v_status, v_note = "WRONG_H", f"order in data but H={data_lvl}≠{target_h}"
                t_v = time.time() - t0
                print(f"    verify: {v_status}  {v_note}  ({t_v*1000:.1f}ms)")
                img = save_image_height(cfg["_img_dir"], label, ordered, target_h, data_lvl, v_status, [])
                results.append({**base, "plan_status": "SUCCESS", "sim_h": data_lvl,
                                 "success": v_status == "VERIFIED", "status": v_status,
                                 "verify_note": v_note, "t_sim_sec": round(t_v, 3),
                                 "image_path": img}); continue

            t0 = time.time()
            df_sim, err, rgb = run_sim(ordered, label, cfg["_exp_dir"])
            if err:
                sim_h, is_coll = None, True
            else:
                step_bboxes, is_coll = parse_bbox_df(df_sim)
                sim_h = None if is_coll else compute_levels(step_bboxes, z_thr)
            t_sim = time.time() - t0; t_sim_total += t_sim
            sim_status = "COLLAPSE" if is_coll else ("SUCCESS" if sim_h == target_h else "FAIL")
            print(f"    sim_h={sim_h}  target={target_h}  {sim_status}  ({t_sim:.1f}s)")
            img = save_image_height(cfg["_img_dir"], label, ordered, target_h, sim_h, sim_status, rgb)
            print(f"    image -> {img}")
            results.append({**base, "plan_status": "SUCCESS", "sim_h": sim_h,
                             "success": sim_status == "SUCCESS", "status": sim_status,
                             "t_sim_sec": round(t_sim, 2), "image_path": img})

    t_total = time.time() - t_start
    df_r = pd.DataFrame(results)
    df_r.to_csv(cfg["_results"], index=False)
    print_summary(df_r, f"tallest/shortest  N={min_n}..{max_n}  simulate={simulate}",
                  min_n, max_n, simulate, t_total, t_plan_total, t_sim_total, cfg, mode="height")

if __name__ == "__main__":
    main()