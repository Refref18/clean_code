"""
plan_height.py  —  shortest / tallest evaluation on real dataset prefix cases.
For each prefix of length N in [min_objects..max_objects]:
  - compute min/max achievable heights across all towers with the same object pool
  - skip if min == max (no variance)
  - plan TWICE: once for min height (shortest), once for max height (tallest)
  - simulate and compare (serial, inline — like plan_task_clean.py)

Usage:  python plan_height.py -c configs/eval_height.yaml
"""
import argparse, ast, os, re, subprocess, sys, random, time
from collections import defaultdict, Counter

import numpy as np, pandas as pd, torch, yaml
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import load_ckpt
from symbol_semantics_fast import get_semantic_symbols
from data_collection_direct.exp import get_experiment
import load_data as _load_data
from load_data import set_seed

# ── config ────────────────────────────────────────────────────────────────────
def load_config(path):
    with open(path) as f: cfg = yaml.safe_load(f)
    root = os.path.join("save", cfg["model_name"], "eval_height")
    cfg.update(_domain   = os.path.join("save", cfg["model_name"], "domain.pddl"),
               _pddl_dir = os.path.join(root, "pddl"),
               _plan_file= os.path.join(root, "pddl", "sas_plan"),
               _exp_dir  = os.path.join(root, "sim"),
               _img_dir  = os.path.join(root, "images"),
               _results  = os.path.join(root, "results.csv"))
    return cfg

# ── helpers ───────────────────────────────────────────────────────────────────
def load_csv(p):
    for enc in ["utf-8","cp1254","latin-1"]:
        try: return pd.read_csv(p, encoding=enc)
        except Exception: pass
    sys.exit(f"Cannot read {p}")

def canonical(t, s):
    try: float(s); return (str(t), str(s))
    except: return (str(t), os.path.basename(str(s)).replace(".urdf",""))

def clean(o): c=canonical(*o); return f"{c[0]}_{c[1]}"

def find_csv(p):
    for q in [p, p.replace(".csv","worker.csv")]:
        if os.path.exists(q): return q
    return None

def row_collapsed(row):
    v = row.get("collapse",0)
    try: return float(v)>=0.5
    except: return str(v).strip().lower() in ("true","1","yes")

# ── object symbols ────────────────────────────────────────────────────────────
def depth_img(otype, opath, depth_dir, device):
    base = f"{otype}_{canonical(otype, opath)[1]}"
    up = os.path.join(depth_dir, f"upper_{base}_new.npy")
    lo = os.path.join(depth_dir, f"lower_{base}_new.npy")
    if os.path.exists(up) and os.path.exists(lo):
        u = torch.from_numpy(np.load(up)).unsqueeze(0).float()
        l = torch.from_numpy(np.load(lo)).unsqueeze(0).float()
        return torch.cat([u,l],0).to(device)
    print(f"  [WARN] depth images missing for {base}")
    return torch.zeros(2,64,64,device=device)

def get_obj_symbols(objects, model, cfg):
    dev = next(model.parameters()).device
    model.eval(); model.gs_obj_layer.hard = model.gs_obj_layer.deterministic = True
    results = []
    with torch.no_grad():
        for ot, op in objects:
            img = depth_img(ot, op, cfg["data"]["depth_image_dir"], torch.device("cpu"))
            b   = next(iter(DataLoader([Data(x=img.unsqueeze(0),
                          edge_index=torch.empty((2,0),dtype=torch.long),
                          edge_attr =torch.empty((0,5),dtype=torch.float))], batch_size=1)))
            f   = model.image_encoder(b.x.float().to(dev))
            sim = F.normalize(f,dim=-1) @ F.normalize(model.cluster_centroids,dim=-1).T
            results.append(model.cluster_codes[sim.argmax(dim=-1)].squeeze(0).cpu().int().tolist())
    return results

# ── collapse symbols ──────────────────────────────────────────────────────────
def get_collapse_syms(model, cfg):
    rel_dim = cfg["_train_cfg"]["model"]["symbol_size"]
    dev     = next(model.parameters()).device
    syms, _, _ = get_semantic_symbols(model, sym_size=rel_dim, out_dim=4,
                                      collapse_threshold=0.5, device=dev)
    return [list(s) for s in syms]

# ── height computation ────────────────────────────────────────────────────────
def compute_levels_stepwise(step_bboxes, threshold):
    levels = 0
    for s in sorted(step_bboxes):
        bbox = step_bboxes[s]
        if not bbox: continue
        new_idx = s - 1
        if new_idx not in bbox: new_idx = max(bbox.keys())
        new_zmax = bbox[new_idx]["max"][2]
        if levels == 0:
            levels = 1
        else:
            prev = [bbox[i]["max"][2] for i in bbox if i != new_idx]
            if prev and new_zmax > max(prev) + threshold:
                levels += 1
    return levels

def tower_levels(rows, z_thr):
    """Compute step-wise levels for one tower's rows. Returns None if collapsed."""
    step_bboxes = {}
    for row in rows:
        if row_collapsed(row): return None
        step = int(row.get("step", 0))
        raw  = row.get("bbox")
        if pd.isna(raw) if isinstance(raw, float) else False: continue
        try:
            bbox = ast.literal_eval(str(raw)) if not isinstance(raw, dict) else raw
            if bbox: step_bboxes[step] = {int(k): v for k,v in bbox.items()}
        except Exception: pass
    return compute_levels_stepwise(step_bboxes, z_thr) if step_bboxes else None

# ── build cases ───────────────────────────────────────────────────────────────
def build_cases(df, min_n, max_n, z_thr):
    """
    Returns (cases, verify_index) where:
      verify_index[(pool, n)] = {canonical_order_tuple: level, ...}
    Used for simulate=false: check if planned order matches a known sequence.
    """
    by_tower = defaultdict(list)
    for _, row in df.iterrows(): by_tower[row["id"]].append(row.to_dict())
    for tid in by_tower: by_tower[tid].sort(key=lambda r: int(r["step"]))

    # build full verification index across ALL towers (not just cases)
    verify_index = defaultdict(dict)  # (pool, n) -> {order_tuple: level}
    for tid, rows in by_tower.items():
        for n in range(min_n, max_n + 1):
            if len(rows) < n: continue
            prefix = rows[:n]
            lvl = tower_levels(prefix, z_thr)
            if lvl is None: continue
            objs  = tuple(canonical(str(r["object_type"]), str(r["object_size_or_path"]))
                          for r in prefix)
            pool  = tuple(sorted(objs))
            verify_index[(pool, n)][objs] = lvl

    cases = []
    for n in range(min_n, max_n + 1):
        by_pool = defaultdict(list)
        for tid, rows in by_tower.items():
            if len(rows) < n: continue
            prefix = rows[:n]
            lvl = tower_levels(prefix, z_thr)
            if lvl is None: continue
            objs = [(str(r["object_type"]), str(r["object_size_or_path"])) for r in prefix]
            pool = tuple(sorted(canonical(*o) for o in objs))
            by_pool[pool].append((tid, objs, lvl))

        for pool, entries in by_pool.items():
            lvls = [lvl for _, _, lvl in entries]
            min_h, max_h = min(lvls), max(lvls)
            if min_h == max_h: continue
            min_entry = min(entries, key=lambda x: x[2])
            max_entry = max(entries, key=lambda x: x[2])
            _, rep_objs, _ = entries[0]
            rep_objs_sorted = sorted(rep_objs, key=lambda o: canonical(*o))
            cases.append({
                "n_objects": n,
                "pool":      pool,
                "objs":      rep_objs_sorted,
                "min_h":     min_h,
                "max_h":     max_h,
                "min_tower": min_entry[0],
                "max_tower": max_entry[0],
                "n_towers":  len(entries),
            })
    return cases, verify_index

# ── PDDL ──────────────────────────────────────────────────────────────────────
def bits_pred(bits, v1, v2):
    return "(and "+" ".join(f"(r{i} {v1} {v2})" if b==1 else f"(not (r{i} {v1} {v2}))" for i,b in enumerate(bits))+")"

def write_problem(obj_syms, rel_dim, obj_dim, target_h, path):
    n     = len(obj_syms)
    usage = f"(active-count-{n})" if n < 4 else "(all-used)"
    def obj_init():
        out = ""
        for i, bits in enumerate(obj_syms):
            out += f"\t\t(top-0 obj{i})\n\t\t"
            out += " ".join(f"(z{j} obj{i})" if v==1 else f"(not (z{j} obj{i}))" for j,v in enumerate(bits))
            out += f" (not (z{obj_dim} obj{i}))\n"
        return out
    def rel_init():
        out = ""
        for i in range(n):
            for j in range(n):
                if i==j: continue
                out += "\t\t"+" ".join(f"(not (r{k} obj{i} obj{j}))" for k in range(rel_dim+1))+"\n"
        return out
    prob = (f"(define (problem blocks-problem)\n\t(:domain blocks)\n"
            f"\t(:objects\n\t\t{' '.join(f'obj{i}' for i in range(n))} - object\n\t)\n"
            f"\t(:init\n\t\t(H0)\n\t\t(active-count-0)\n{obj_init()}{rel_init()}\t)\n"
            f"\t(:goal (and\n\t\t(H{target_h})\n\t\t{usage}\n"
            f"\t\t(not (active-count-collapse))\n\t))\n)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(prob)

# ── planner ───────────────────────────────────────────────────────────────────
def run_planner(problem_path, cfg):
    import signal as _signal
    pf  = cfg["_plan_file"]
    t   = int(cfg["planner"].get("time_limit_sec", 10))
    if os.path.exists(pf): os.remove(pf)
    cmd = [sys.executable, cfg["planner"]["fast_downward"],
           "--plan-file", pf,
           cfg["_domain"], problem_path,
           "--search", cfg["planner"]["search"]]
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                preexec_fn=os.setsid)
        proc.wait(timeout=t)
    except subprocess.TimeoutExpired:
        if proc:
            try: os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
            except ProcessLookupError: pass
        return "TIMEOUT", []
    except Exception as e:
        return f"ERROR_{e}", []

    if os.path.exists(pf):
        lines = [l.strip() for l in open(pf) if l.strip() and not l.startswith(";")]
        return "SUCCESS", lines
    return "NO_PLAN", []

def plan_order(lines, objects):
    placed = []
    for line in lines:
        toks = line.lstrip("(").rstrip(")").split()
        if not toks or not toks[0].startswith("a_"): continue
        obj_t = [t for t in toks[1:] if re.match(r"^obj\d+$", t)]
        if obj_t:
            idx = int(obj_t[-1][3:])
            if idx < len(objects): placed.append(objects[idx])
    return placed

# ── simulation (inline, serial — mirrors plan_task_clean.py) ─────────────────
def _find_existing_csv(path):
    for p in [path,
              path.replace(".csv","worker.csv"),
              os.path.join(os.path.dirname(path),
                           os.path.basename(path).replace(".csv","")+"worker.csv")]:
        if os.path.exists(p): return p
    return None

def run_sim(ordered, run_label, exp_dir, z_thr):
    """Run get_experiment inline and return (sim_levels, collapsed, rgb_paths)."""
    try:
        get_experiment(ordered, exp_dir+"/", run_label)
    except Exception as e:
        print(f"  [SIM ERROR] {e}")
        return None, True, []

    csv_path = os.path.join(exp_dir, f"{run_label}.csv")
    actual   = _find_existing_csv(csv_path)
    if actual is None:
        print(f"  [SIM ERROR] CSV not found: {csv_path}")
        return None, True, []
    try:
        df = load_csv(actual)
    except Exception as e:
        print(f"  [SIM ERROR] Cannot read CSV: {e}")
        return None, True, []

    step_bboxes, is_coll = {}, False
    for _, row in df.iterrows():
        if row_collapsed(row): is_coll = True
        step = int(row.get("step", 0))
        raw  = row.get("bbox")
        if raw is None or (isinstance(raw, float) and pd.isna(raw)): continue
        try:
            bbox = ast.literal_eval(str(raw)) if not isinstance(raw, dict) else raw
            if bbox: step_bboxes[step] = {int(k): v for k,v in bbox.items()}
        except Exception:
            pass

    if is_coll: return None, True, []
    lvl = compute_levels_stepwise(step_bboxes, z_thr)
    rgb = []
    for _, row in df.sort_values("step").iterrows():
        p = str(row.get("rgb_image_path",""))
        if p and p != "nan" and os.path.exists(p): rgb.append(p)
    return lvl, False, rgb

# ── image ─────────────────────────────────────────────────────────────────────
def save_image(cfg, label, ordered, target_h, sim_h, status, img_paths):
    os.makedirs(cfg["_img_dir"], exist_ok=True)
    IH, IW, CH = 256, 320, 100
    frames = []
    for p in img_paths:
        try: frames.append(Image.open(p).convert("RGB").resize((IW,IH),Image.LANCZOS))
        except: pass
    if not frames: frames = [Image.new("RGB",(IW,IH),(40,40,40))]
    canvas = Image.new("RGB",(IW*len(frames),IH+CH),(15,15,15))
    draw   = ImageDraw.Draw(canvas)
    for i,im in enumerate(frames):
        canvas.paste(im,(i*IW,0)); draw.text((i*IW+4,4),f"step {i+1}",fill=(255,255,255))
    try:
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",13)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",11)
    except: fb = fs = ImageFont.load_default()
    color = {"SUCCESS":(80,200,120),"FAIL":(220,80,60),
             "COLLAPSE":(180,80,220),"NO_PLAN":(255,165,0),
             "VERIFIED":(80,200,120),"WRONG_H":(220,80,60),
             "UNVERIFIED":(180,180,60)}.get(status,(200,200,200))
    y = IH+4
    draw.text((8,y),f"{label}  {status}  target=H{target_h}  sim=H{sim_h}",fill=color,font=fb); y+=18
    draw.text((8,y),(" > ".join(clean(o) for o in ordered))[:120],fill=(150,150,150),font=fs)
    safe = re.sub(r"[^\w\-]","_",f"{label}_{status}_H{target_h}")
    p = os.path.join(cfg["_img_dir"],f"{safe}.png"); canvas.save(p); return p

# ── sampling ─────────────────────────────────────────────────────────────────
def _sample_diverse(all_cases, n_per_n):
    """
    Per N: pick up to n_per_n cases prioritising unique pools.
    If fewer unique pools than n_per_n, fill by cycling pools again.
    Each case already represents one unique pool, so first pass takes all,
    then we repeat from the same pool list until cap is reached.
    """
    by_n = defaultdict(list)
    for c in all_cases: by_n[c["n_objects"]].append(c)
    result = []
    for n in sorted(by_n):
        pool_cases = by_n[n]; random.shuffle(pool_cases)
        selected = list(pool_cases)           # one case per pool already
        # fill remaining slots by repeating pools in shuffled order
        idx = 0
        while len(selected) < n_per_n and pool_cases:
            selected.append(pool_cases[idx % len(pool_cases)])
            idx += 1
        result.extend(selected[:n_per_n])
    return result

# ── summary ───────────────────────────────────────────────────────────────────
def _print_summary_height(df_r, min_n, max_n, simulate, t_total, t_plan_total, t_sim_total, cfg):
    W = 100
    print(f"\n{'='*W}")
    print(f"  SUMMARY  tallest/shortest  N={min_n}..{max_n}  simulate={simulate}")
    print(f"{'='*W}")

    def _pct(num, den): return f"{100*num/den:.1f}%" if den else "—"

    if simulate:
        hdr = (f"{'N':>2}  {'goal':<5}  {'total':>5}  {'no_plan':>7}  {'no_plan%':>8}  "
               f"{'success':>7}  {'success%':>8}  {'fail':>4}  {'fail%':>5}  "
               f"{'collapse':>8}  {'collapse%':>9}  {'t_plan_avg':>10}  {'t_sim_avg':>9}")
    else:
        hdr = (f"{'N':>2}  {'goal':<5}  {'total':>5}  {'no_plan':>7}  {'no_plan%':>8}  "
               f"{'planned':>7}  {'planned%':>8}  {'timeout':>7}  {'timeout%':>8}  "
               f"{'t_plan_avg':>10}")
    print(hdr)
    print("-" * W)

    totals = dict(total=0, no_plan=0, success=0, fail=0, collapse=0, timeout=0, planned=0,
                  t_plan=0.0, t_sim=0.0)

    for n in range(min_n, max_n + 1):
        sub = df_r[df_r["n_objects"] == n]
        if sub.empty: continue
        for tag in ["short", "tall"]:
            s = sub[sub["goal"] == tag]
            if s.empty: continue
            tot     = len(s)
            no_plan = (s["status"] == "NO_PLAN").sum()
            timeout = (s["plan_status"] == "TIMEOUT").sum()
            planned = (s["plan_status"] == "SUCCESS").sum()
            t_pa    = s["t_plan_sec"].mean() if "t_plan_sec" in s else 0.0
            totals["total"]   += tot
            totals["no_plan"] += int(no_plan)
            totals["timeout"] += int(timeout)
            totals["planned"] += int(planned)
            totals["t_plan"]  += s["t_plan_sec"].sum() if "t_plan_sec" in s else 0.0

            if simulate:
                success  = (s["status"] == "SUCCESS").sum()
                fail     = (s["status"] == "FAIL").sum()
                collapse = (s["status"] == "COLLAPSE").sum()
                t_sa     = s["t_sim_sec"].mean() if "t_sim_sec" in s else 0.0
                totals["success"]  += int(success)
                totals["fail"]     += int(fail)
                totals["collapse"] += int(collapse)
                totals["t_sim"]    += s["t_sim_sec"].sum() if "t_sim_sec" in s else 0.0
                print(f"{n:>2}  {tag:<5}  {tot:>5}  {no_plan:>7}  {_pct(no_plan,tot):>8}  "
                      f"{success:>7}  {_pct(success,tot):>8}  {fail:>4}  {_pct(fail,tot):>5}  "
                      f"{collapse:>8}  {_pct(collapse,tot):>9}  {t_pa:>10.2f}s  {t_sa:>8.2f}s")
            else:
                print(f"{n:>2}  {tag:<5}  {tot:>5}  {no_plan:>7}  {_pct(no_plan,tot):>8}  "
                      f"{planned:>7}  {_pct(planned,tot):>8}  {timeout:>7}  {_pct(timeout,tot):>8}  "
                      f"{t_pa:>10.2f}s")

    print("-" * W)
    T = totals["total"]
    if simulate:
        t_pa = totals["t_plan"] / max(T, 1)
        t_sa = totals["t_sim"]  / max(T, 1)
        print(f"{'':>2}  {'TOTAL':<5}  {T:>5}  {totals['no_plan']:>7}  "
              f"{_pct(totals['no_plan'],T):>8}  "
              f"{totals['success']:>7}  {_pct(totals['success'],T):>8}  "
              f"{totals['fail']:>4}  {_pct(totals['fail'],T):>5}  "
              f"{totals['collapse']:>8}  {_pct(totals['collapse'],T):>9}  "
              f"{t_pa:>10.2f}s  {t_sa:>8.2f}s")
    else:
        t_pa = totals["t_plan"] / max(T, 1)
        print(f"{'':>2}  {'TOTAL':<5}  {T:>5}  {totals['no_plan']:>7}  "
              f"{_pct(totals['no_plan'],T):>8}  "
              f"{totals['planned']:>7}  {_pct(totals['planned'],T):>8}  "
              f"{totals['timeout']:>7}  {_pct(totals['timeout'],T):>8}  "
              f"{t_pa:>10.2f}s")

    print(f"\n── Timing {'─'*40}")
    print(f"  total wall time  : {t_total:.1f}s")
    print(f"  planning total   : {t_plan_total:.1f}s  (avg {t_plan_total/max(T,1):.2f}s/case)")
    if simulate:
        print(f"  simulation total : {t_sim_total:.1f}s  (avg {t_sim_total/max(T,1):.2f}s/case)")
    print(f"  results -> {cfg['_results']}")
    print(f"  images  -> {cfg['_img_dir']}")
    print(f"{'='*W}\n")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config",default="configs/eval_height.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["eval"].get("seed",42))); random.seed(int(cfg["eval"].get("seed",42)))
    for d in [cfg["_pddl_dir"],cfg["_exp_dir"],cfg["_img_dir"]]: os.makedirs(d,exist_ok=True)
    if not os.path.exists(cfg["_domain"]): sys.exit(f"domain.pddl not found: {cfg['_domain']}")

    # model + symbols
    model, train_cfg = load_ckpt(cfg["model_name"], tag="best")
    cfg["_train_cfg"] = train_cfg; _load_data.init(train_cfg); set_seed(train_cfg["seed"])
    model.freeze(deterministic=True)
    rel_dim = train_cfg["model"]["symbol_size"]; obj_dim = train_cfg["model"]["obj_symbol_size"]
    collapse_syms = get_collapse_syms(model, cfg)
    print(f"  collapse symbols: {collapse_syms}")

    # build cases
    min_n = int(cfg["eval"].get("min_objects", 2))
    max_n = int(cfg["eval"].get("max_objects", 4))
    z_thr = float(cfg["eval"].get("z_level_threshold", 0.15))
    df    = load_csv(cfg["data"]["dataset_csv"])
    print(f"\n--> Loaded {len(df)} rows. Building cases N={min_n}..{max_n}...")

    all_cases, verify_index = build_cases(df, min_n, max_n, z_thr)
    print(f"  {len(all_cases)} cases with min≠max height")
    per_n = Counter(c["n_objects"] for c in all_cases)
    for n in sorted(per_n): print(f"    N={n}: {per_n[n]} pools")

    n_cap = int(cfg["eval"].get("n_scenarios_per_n", 0))
    if n_cap > 0:
        all_cases = _sample_diverse(all_cases, n_cap)
        print(f"  After sampling {n_cap}/N: {len(all_cases)} cases")

    # object symbol cache
    unique_objs = list({canonical(*o) for c in all_cases for o in c["objs"]})
    syms        = get_obj_symbols(unique_objs, model, cfg)
    sym_cache   = {o: s for o,s in zip(unique_objs, syms)}
    print(f"  {len(sym_cache)} unique object symbols cached\n")

    simulate = bool(cfg["eval"].get("simulate", True))

    # ── plan + simulate inline (serial, like plan_task_clean.py) ─────────────
    print(f"\n── Planning & {'Simulating' if simulate else 'Verifying'} "
          f"({len(all_cases)} pools × 2 heights) ──")
    results = []
    t_plan_total = 0.0
    t_sim_total  = 0.0
    t_start      = time.time()

    for i, case in enumerate(all_cases):
        objs     = case["objs"]
        obj_syms = [sym_cache.get(canonical(*o)) for o in objs]
        if any(s is None for s in obj_syms):
            print(f"  [{i}] SKIP missing symbol"); continue

        for target_h, goal_tag in [(case["min_h"],"short"), (case["max_h"],"tall")]:
            label     = f"{i:04d}_N{case['n_objects']}_{goal_tag}_H{target_h}"
            prob_path = os.path.join(cfg["_pddl_dir"], f"p_{label}.pddl")
            write_problem(obj_syms, rel_dim, obj_dim, target_h, prob_path)

            t0 = time.time()
            status, lines = run_planner(prob_path, cfg)
            t_plan = time.time() - t0
            t_plan_total += t_plan
            print(f"  [{i}] N={case['n_objects']}  H{target_h} ({goal_tag})  "
                  f"planner={status}  ({t_plan:.1f}s)")

            row = dict(pool=str(case["pool"]), n_objects=case["n_objects"], goal=goal_tag,
                       target_h=target_h, min_h=case["min_h"], max_h=case["max_h"],
                       t_plan_sec=round(t_plan, 2))

            if status != "SUCCESS":
                row.update(plan_status=status, plan_order="", sim_h=None,
                           success=False, status="NO_PLAN", t_sim_sec=0.0,
                           image_path=save_image(cfg, label, objs, target_h, None, "NO_PLAN", []))
                results.append(row); continue

            ordered = plan_order(lines, objs)
            row["plan_order"] = " > ".join(clean(o) for o in ordered)
            print(f"    order: {row['plan_order']}")

            if not simulate:
                t0 = time.time()
                order_key = tuple(canonical(*o) for o in ordered)
                known     = verify_index.get((case["pool"], case["n_objects"]), {})
                data_lvl  = known.get(order_key)
                if data_lvl is None:
                    achieved = [lvl for lvl in known.values() if lvl == target_h]
                    v_status = "UNVERIFIED"
                    v_note   = f"order unseen in data; {len(achieved)} known seqs hit H{target_h}"
                elif data_lvl == target_h:
                    v_status = "VERIFIED"
                    v_note   = f"exact order found in data with H={data_lvl}"
                else:
                    v_status = "WRONG_H"
                    v_note   = f"order found in data but H={data_lvl} ≠ target={target_h}"
                t_v = time.time() - t0
                print(f"    data-verify: {v_status}  ({v_note})  ({t_v*1000:.1f}ms)")
                img = save_image(cfg, label, ordered, target_h, data_lvl, v_status, [])
                row.update(plan_status="SUCCESS", sim_h=data_lvl, success=(v_status=="VERIFIED"),
                           status=v_status, verify_note=v_note, t_sim_sec=round(t_v,3),
                           image_path=img)
                print(f"    image -> {img}")
                results.append(row); continue

            # inline simulation
            t0 = time.time()
            sim_h, is_coll, rgb = run_sim(ordered, label, cfg["_exp_dir"], z_thr)
            t_sim = time.time() - t0
            t_sim_total += t_sim
            sim_status = ("COLLAPSE" if is_coll else
                          "SUCCESS"  if sim_h == target_h else "FAIL")
            print(f"    sim_h={sim_h}  target={target_h}  {sim_status}  ({t_sim:.1f}s)")
            img = save_image(cfg, label, ordered, target_h, sim_h, sim_status, rgb)
            print(f"    image -> {img}")
            row.update(plan_status="SUCCESS", sim_h=sim_h,
                       success=(sim_status=="SUCCESS"), status=sim_status,
                       t_sim_sec=round(t_sim, 2), image_path=img)
            results.append(row)

    t_total = time.time() - t_start

    df_r = pd.DataFrame(results)
    df_r.to_csv(cfg["_results"], index=False)

    _print_summary_height(df_r, min_n, max_n, simulate, t_total, t_plan_total, t_sim_total, cfg)

if __name__ == "__main__":
    main()