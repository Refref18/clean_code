"""
plan_task_fast.py — inside / occlude task evaluation.
Usage:  python plan_task_fast.py -c configs/eval_task.yaml
"""
import argparse, ast, os, random, sys, time
from collections import defaultdict, Counter

import pandas as pd, yaml

from eval_utils import (canonical, clean, load_csv, load_model, get_obj_symbols,
                        get_collapse_syms, run_planner, plan_order, run_sim,
                        obj_init_pddl, rel_init_pddl, bits_pred,
                        save_image_task, sample_diverse, print_summary, row_collapsed)
from symbol_semantics_fast import get_semantic_symbols, get_symbol_map
from load_data import set_seed

# ── config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f: cfg = yaml.safe_load(f)
    root = os.path.join("save", cfg["model_name"], "eval_task", cfg["eval"]["task"])
    cfg.update(_domain   =os.path.join("save", cfg["model_name"], "domain.pddl"),
               _pddl_dir =os.path.join(root, "pddl"),
               _plan_file=os.path.join(root, "pddl", "sas_plan"),
               _exp_dir  =os.path.join(root, "sim"),
               _img_dir  =os.path.join(root, "images"),
               _results  =os.path.join(root, "results.csv"))
    return cfg

# ── task symbols ──────────────────────────────────────────────────────────────

def get_task_syms(model, train_cfg, task, eps=0.0):
    rel_dim = train_cfg["model"]["symbol_size"]
    dev     = next(model.parameters()).device
    _, inserted, _ = get_semantic_symbols(model, sym_size=rel_dim, out_dim=4,
                                          collapse_threshold=0.5, device=dev)
    sym_map = get_symbol_map(model, sym_size=rel_dim, out_dim=4, device=dev)
    # width = MaxX - MinX: negative → inside, positive → occlude
    scored  = sorted([(list(s), (sym_map[tuple(float(b) for b in s)][2] -
                                 sym_map[tuple(float(b) for b in s)][0]).item())
                      for s in inserted], key=lambda x: x[1])
    if task == "inside":   sel = [s for s,w in scored if w < -eps] or [scored[0][0]]
    elif task == "occlude": sel = [s for s,w in scored if w >  eps] or [scored[-1][0]]
    else: raise ValueError(f"Unknown task: {task}")
    print(f"  task symbols ({task}): {sel}")
    return sel

# ── build cases ───────────────────────────────────────────────────────────────

def _task_pairs_in(steps, task):
    pl = {int(r["step"])-1: (str(r["object_type"]), str(r["object_size_or_path"])) for r in steps}
    pairs = []
    for row in steps:
        step = int(row["step"])
        if step == 1: continue
        raw = row.get("bounding_box_differences")
        if raw is None or (isinstance(raw, float) and pd.isna(raw)): continue
        try: diff = ast.literal_eval(str(raw))
        except: continue
        for k, entry in diff.items():
            if not isinstance(entry, dict): continue
            if task not in str(entry.get("situation","")).lower(): continue
            ei = int(k)
            if task == "inside" and pl.get(ei, ("",))[0].lower() != "cup": continue
            pairs.append({"new_obj": pl.get(step-1), "existing_obj": pl.get(ei)})
    return pairs

def build_cases(df, task, min_n, max_n):
    by_tower = defaultdict(list)
    for _, row in df.iterrows(): by_tower[row["id"]].append(row.to_dict())
    verify_index = defaultdict(set)
    cases = []
    for tid, rows in by_tower.items():
        rows.sort(key=lambda r: int(r["step"]))
        for n in range(min_n, max_n + 1):
            if len(rows) < n: continue
            prefix = rows[:n]
            if any(row_collapsed(r) for r in prefix): continue
            pairs = _task_pairs_in(prefix, task)
            if not pairs: continue
            objs = tuple(canonical(str(r["object_type"]), str(r["object_size_or_path"])) for r in prefix)
            pool = tuple(sorted(objs))
            verify_index[(pool, n)].add(objs)
            cases.append(dict(tower_id=tid, n_objects=n, steps=prefix, exp_pairs=pairs, pool=pool))
    return cases, verify_index

# ── PDDL problem ──────────────────────────────────────────────────────────────

def write_problem(obj_syms, exp_pairs, task_syms, sym_cache, rel_dim, obj_dim, path, goal_mode):
    n     = len(obj_syms)
    usage = f"(active-count-{n})" if n < 4 else "(all-used)"

    if goal_mode == "any":
        goal = "\t\t(or\n" + "".join(
            f"\t\t\t(and {bits_pred(ts, f'obj{i}', f'obj{j}')} (r{rel_dim} obj{i} obj{j}) )\n"
            for i in range(n) for j in range(n) if i != j for ts in task_syms
        ) + "\t\t)\n"
    else:
        goal = ""
        for ep in exp_pairs:
            es = sym_cache.get(canonical(*ep["existing_obj"]))
            ns = sym_cache.get(canonical(*ep["new_obj"]))
            if es is None or ns is None: continue
            goal += "\t\t(or\n" + "".join(
                f"\t\t\t(and {bits_pred(ts, f'obj{ei}', f'obj{ni}')} (r{rel_dim} obj{ei} obj{ni}) )\n"
                for ei, s in enumerate(obj_syms) if s == es
                for ni, s2 in enumerate(obj_syms) if s2 == ns and ni != ei
                for ts in task_syms
            ) + "\t\t)\n"

    prob = (f"(define (problem blocks-problem)\n\t(:domain blocks)\n"
            f"\t(:objects\n\t\t{' '.join(f'obj{i}' for i in range(n))} - object\n\t)\n"
            f"\t(:init\n\t\t(H0)\n\t\t(active-count-0)\n"
            f"{obj_init_pddl(obj_syms, obj_dim)}{rel_init_pddl(n, rel_dim)}\t)\n"
            f"\t(:goal (and\n\t\t{usage}\n\t\t(not (active-count-collapse))\n{goal}\t))\n)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(prob)

# ── simulation ────────────────────────────────────────────────────────────────

def sim_task(ordered, run_label, cfg):
    """Simulate; return (collapsed, found_pairs, rgb_paths)."""
    task = cfg["eval"]["task"]
    df, err, rgb = run_sim(ordered, f"{task}_{run_label}", cfg["_exp_dir"])
    if err: return True, [], []

    is_coll = False
    for _, row in df.iterrows():
        if row_collapsed(row): is_coll = True; break
    if is_coll: return True, [], rgb

    found = []
    for _, row in df.iterrows():
        step = int(row.get("step", 0)); raw = row.get("bounding_box_differences")
        if raw is None or (isinstance(raw, float) and pd.isna(raw)): continue
        try: diff = ast.literal_eval(str(raw))
        except: continue
        for k, entry in diff.items():
            if not isinstance(entry, dict): continue
            if task not in str(entry.get("situation","")).lower(): continue
            ei = int(k); new_idx = step - 1
            found.append({"new_obj":      ordered[new_idx] if new_idx < len(ordered) else None,
                          "existing_obj": ordered[ei]      if ei      < len(ordered) else None})
    return False, found, rgb

def compute_success(expected, found):
    found_set = {(canonical(*fp["new_obj"]), canonical(*fp["existing_obj"]))
                 for fp in found if fp.get("new_obj") and fp.get("existing_obj")}
    matched = sum(1 for ep in expected
                  if ep.get("new_obj") and ep.get("existing_obj")
                  and (canonical(*ep["new_obj"]), canonical(*ep["existing_obj"])) in found_set)
    return matched, 100.0 * matched / len(expected) if expected else 0.0

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/eval_task.yaml")
    args = parser.parse_args()

    cfg  = load_config(args.config); task = cfg["eval"]["task"]
    seed = int(cfg["eval"].get("seed", 42))
    set_seed(seed); random.seed(seed)
    for d in [cfg["_pddl_dir"], cfg["_exp_dir"], cfg["_img_dir"]]: os.makedirs(d, exist_ok=True)
    if not os.path.exists(cfg["_domain"]): sys.exit(f"domain.pddl not found: {cfg['_domain']}")

    model, train_cfg = load_model(cfg)
    rel_dim   = train_cfg["model"]["symbol_size"]
    obj_dim   = train_cfg["model"]["obj_symbol_size"]
    task_syms = get_task_syms(model, train_cfg, task, float(cfg["eval"].get("width_epsilon", 0.0)))

    min_n = int(cfg["eval"].get("min_objects", 2))
    max_n = int(cfg["eval"].get("max_objects", 4))
    df    = load_csv(cfg["data"]["dataset_csv"])
    print(f"\n→ {len(df)} rows. Building prefix cases N={min_n}..{max_n}...")

    all_cases, verify_index = build_cases(df, task, min_n, max_n)
    per_n = Counter(c["n_objects"] for c in all_cases)
    print(f"  {len(all_cases)} cases  {dict(sorted(per_n.items()))}")

    n_cap = int(cfg["eval"].get("n_baseline_scenarios", 0))
    if n_cap > 0:
        all_cases = sample_diverse(all_cases, n_cap)
        print(f"  After sampling: {len(all_cases)} cases")

    unique_objs = list({canonical(str(r["object_type"]), str(r["object_size_or_path"]))
                        for c in all_cases for r in c["steps"]})
    sym_cache   = dict(zip(unique_objs, get_obj_symbols(unique_objs, model, cfg["data"]["depth_image_dir"])))
    print(f"  {len(sym_cache)} unique object symbols cached\n")

    simulate  = bool(cfg["eval"].get("simulate", True))
    goal_mode = str(cfg["eval"].get("goal_mode", "specific")).lower()
    print(f"  goal_mode: {goal_mode}\n")

    results, t_plan_total, t_sim_total, t_start = [], 0.0, 0.0, time.time()

    for i, case in enumerate(all_cases):
        tid      = case["tower_id"]; n_objs = case["n_objects"]
        exp_pairs = case["exp_pairs"]
        objs     = [(str(r["object_type"]), str(r["object_size_or_path"]))
                    for r in sorted(case["steps"], key=lambda r: int(r["step"]))]
        obj_syms = [sym_cache.get(canonical(*o)) for o in objs]
        if any(s is None for s in obj_syms): print(f"  [{i}] SKIP — missing symbol"); continue

        prob_path = os.path.join(cfg["_pddl_dir"], f"p_{i:04d}_N{n_objs}_{tid}.pddl")
        write_problem(obj_syms, exp_pairs, task_syms, sym_cache, rel_dim, obj_dim, prob_path, goal_mode)

        t0 = time.time()
        status, lines = run_planner(prob_path, cfg)
        t_plan = time.time() - t0; t_plan_total += t_plan
        print(f"[{i+1}/{len(all_cases)}] tower={tid} N={n_objs} goal={goal_mode} {status} ({t_plan:.1f}s)")

        base = dict(tower_id=tid, n_objects=n_objs, plan_status=status,
                    goal_mode=goal_mode, n_expected=len(exp_pairs), t_plan_sec=round(t_plan, 2))

        if status != "SUCCESS":
            img = save_image_task(cfg["_img_dir"], i, objs, exp_pairs, [], 0.0, "NO_PLAN", [], task)
            results.append({**base, "n_matched":0, "pct":0.0, "status":"NO_PLAN",
                             "t_sim_sec":0.0, "image_path":img}); continue

        ordered = plan_order(lines, objs)
        print(f"  order: {' > '.join(clean(o) for o in ordered)}")

        if not simulate:
            t0 = time.time()
            order_key = tuple(canonical(*o) for o in ordered)
            known     = verify_index.get((case["pool"], n_objs), set())
            v_status  = "VERIFIED" if order_key in known else "UNVERIFIED"
            v_note    = ("exact order found" if v_status == "VERIFIED"
                         else f"not in data; {len(known)} known seqs achieve task")
            n_matched, pct = (len(exp_pairs), 100.0) if v_status == "VERIFIED" else (0, 0.0)
            t_v = time.time() - t0
            print(f"  verify: {v_status}  {v_note}  ({t_v*1000:.1f}ms)")
            img = save_image_task(cfg["_img_dir"], i, ordered, exp_pairs, [], pct, v_status, [], task)
            print(f"  image -> {img}")
            results.append({**base, "n_matched":n_matched, "pct":pct, "status":v_status,
                             "verify_note":v_note, "t_sim_sec":round(t_v,3), "image_path":img}); continue

        t0 = time.time()
        is_coll, found, rgb = sim_task(ordered, f"{i:04d}_N{n_objs}_{tid}", cfg)
        t_sim = time.time() - t0; t_sim_total += t_sim

        if is_coll:
            n_matched, pct, sim_status = 0, 0.0, "COLLAPSE"
        else:
            n_matched, pct = compute_success(exp_pairs, found)
            sim_status = "SUCCESS" if pct == 100 else "PARTIAL" if pct > 0 else "FAIL"

        print(f"  sim: {sim_status}  {n_matched}/{len(exp_pairs)}  ({pct:.0f}%)  ({t_sim:.1f}s)")
        img = save_image_task(cfg["_img_dir"], i, ordered, exp_pairs, found, pct, sim_status, rgb, task)
        print(f"  image -> {img}")
        results.append({**base, "n_matched":n_matched, "pct":pct, "status":sim_status,
                        "t_sim_sec":round(t_sim,2), "image_path":img})

    t_total = time.time() - t_start
    df_r = pd.DataFrame(results)
    df_r.to_csv(cfg["_results"], index=False)
    print_summary(df_r, f"task={task}  goal={goal_mode}  N={min_n}..{max_n}  simulate={simulate}",
                  min_n, max_n, simulate, t_total, t_plan_total, t_sim_total, cfg, mode="task")

if __name__ == "__main__":
    main()