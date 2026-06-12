"""
plan_task.py  —  inside / occlude evaluation on real dataset prefix cases.
Serial planning + serial simulation (inline, like plan_task_clean.py).

Usage:  python plan_task.py -c configs/eval_task.yaml
"""
import argparse, ast, os, re, subprocess, sys, random, time
from collections import defaultdict, Counter

import numpy as np, pandas as pd, torch, yaml
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import load_ckpt
from symbol_semantics_fast import get_semantic_symbols, get_symbol_map
from data_collection_direct.exp import get_experiment
import load_data as _load_data
from load_data import set_seed

# ── config ────────────────────────────────────────────────────────────────────
def load_config(path):
    with open(path) as f: cfg = yaml.safe_load(f)
    root = os.path.join("save", cfg["model_name"], "eval_task", cfg["eval"]["task"])
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

def _find_csv(p):
    for q in [p, p.replace(".csv","worker.csv")]:
        if os.path.exists(q): return q
    return None

def _row_collapsed(row):
    v = row.get("collapse", 0)
    try: return float(v) >= 0.5
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

# ── relation symbols ──────────────────────────────────────────────────────────
def get_collapse_syms(model, cfg):
    rel_dim = cfg["_train_cfg"]["model"]["symbol_size"]
    dev     = next(model.parameters()).device
    syms,_,_ = get_semantic_symbols(model, sym_size=rel_dim, out_dim=4,
                                    collapse_threshold=0.5, device=dev)
    return [list(s) for s in syms]

def get_task_syms(model, cfg):
    task    = cfg["eval"]["task"]
    rel_dim = cfg["_train_cfg"]["model"]["symbol_size"]
    dev     = next(model.parameters()).device
    eps     = float(cfg["eval"].get("width_epsilon", 0.0))
    _, inserted, _ = get_semantic_symbols(model, sym_size=rel_dim, out_dim=4,
                                          collapse_threshold=0.5, device=dev)
    sym_map = get_symbol_map(model, sym_size=rel_dim, out_dim=4, device=dev)
    scored  = sorted([(list(s),(sym_map[tuple(float(b) for b in s)][2]-
                                sym_map[tuple(float(b) for b in s)][0]).item())
                      for s in inserted], key=lambda x: x[1])
    if task == "inside":  sel = [s for s,w in scored if w < -eps] or [scored[0][0]]
    elif task == "occlude": sel = [s for s,w in scored if w >  eps] or [scored[-1][0]]
    else: raise ValueError(task)
    print(f"  task symbols ({task}): {sel}")
    return sel

# ── dataset: build prefix cases ───────────────────────────────────────────────
def _task_pairs_in(steps, task):
    pl = {int(r["step"])-1: (str(r["object_type"]),str(r["object_size_or_path"])) for r in steps}
    pairs = []
    for row in steps:
        step = int(row["step"])
        if step == 1: continue
        raw = row.get("bounding_box_differences")
        if pd.isna(raw) if isinstance(raw,float) else (raw is None): continue
        try: diff = ast.literal_eval(str(raw))
        except: continue
        for k, entry in diff.items():
            if not isinstance(entry,dict): continue
            if task not in str(entry.get("situation","")).lower(): continue
            ei = int(k)
            if task == "inside" and pl.get(ei,("",))[0].lower() != "cup": continue
            pairs.append({"new_obj": pl.get(step-1), "existing_obj": pl.get(ei)})
    return pairs

def build_cases(df, task, min_n, max_n):
    """
    Returns (cases, verify_index) where:
      verify_index[(pool, n)] = set of canonical_order_tuples that achieved the task
    Used for simulate=false: check if planned order is a known-good sequence.
    """
    by_tower = defaultdict(list)
    for _,row in df.iterrows(): by_tower[row["id"]].append(row.to_dict())

    # full verification index across all towers
    verify_index = defaultdict(set)  # (pool, n) -> {order_tuple, ...}
    for tid, rows in by_tower.items():
        rows.sort(key=lambda r: int(r["step"]))
        for n in range(min_n, max_n+1):
            if len(rows) < n: continue
            prefix = rows[:n]
            if any(_row_collapsed(r) for r in prefix): continue
            if not _task_pairs_in(prefix, task): continue
            objs = tuple(canonical(str(r["object_type"]),str(r["object_size_or_path"]))
                         for r in prefix)
            pool = tuple(sorted(objs))
            verify_index[(pool, n)].add(objs)

    cases = []
    for tid, rows in by_tower.items():
        rows.sort(key=lambda r: int(r["step"]))
        for n in range(min_n, max_n+1):
            if len(rows) < n: continue
            prefix = rows[:n]
            if any(_row_collapsed(r) for r in prefix): continue
            pairs = _task_pairs_in(prefix, task)
            if not pairs: continue
            cases.append({"tower_id": tid, "n_objects": n,
                          "steps": prefix, "exp_pairs": pairs})
    return cases, verify_index

def _sample_diverse(all_cases, n_per_n):
    by_n = defaultdict(list)
    for c in all_cases: by_n[c["n_objects"]].append(c)
    result = []
    for n in sorted(by_n):
        cases = by_n[n]; random.shuffle(cases)
        by_pool = defaultdict(list)
        for c in cases:
            pool = tuple(sorted(canonical(str(r["object_type"]),str(r["object_size_or_path"]))
                                for r in c["steps"]))
            by_pool[pool].append(c)
        pools = list(by_pool.keys()); random.shuffle(pools)
        selected = [by_pool[p].pop() for p in pools]
        while len(selected) < n_per_n:
            added = False
            for p in pools:
                if len(selected) >= n_per_n: break
                if by_pool[p]: selected.append(by_pool[p].pop()); added = True
            if not added: break
        result.extend(selected[:n_per_n])
    return result

# ── PDDL ──────────────────────────────────────────────────────────────────────
def _bits_pred(bits, v1, v2):
    return "(and "+" ".join(f"(r{i} {v1} {v2})" if b==1 else f"(not (r{i} {v1} {v2}))" for i,b in enumerate(bits))+")"

def write_problem(obj_syms, exp_pairs, task_syms, sym_cache, rel_dim, obj_dim, path, goal_mode="specific"):
    n = len(obj_syms)
    usage = f"(active-count-{n})" if n < 4 else "(all-used)"

    def obj_init():
        out = ""
        for i,bits in enumerate(obj_syms):
            out += f"\t\t(top-0 obj{i})\n"
            z_true = " ".join(f"(z{j} obj{i})" for j,v in enumerate(bits) if v == 1)
            if z_true:
                out += f"\t\t{z_true}\n"
            # z{obj_dim} (active bit) intentionally omitted — starts false by closed world
        return out

    def rel_init():
        return ""  # all relations false by closed-world assumption

    def goal_specific():
        # one (or ...) per expected pair, resolved through object symbols
        out = ""
        for ep in exp_pairs:
            es = sym_cache.get(canonical(*ep["existing_obj"]))
            ns = sym_cache.get(canonical(*ep["new_obj"]))
            if es is None or ns is None: continue
            out += "\t\t(or\n"
            for ei,s in enumerate(obj_syms):
                if s != es: continue
                for ni,s2 in enumerate(obj_syms):
                    if s2 != ns or ni == ei: continue
                    for ts in task_syms:
                        out += f"\t\t\t(and {_bits_pred(ts,f'obj{ei}',f'obj{ni}')} (r{rel_dim} obj{ei} obj{ni}) )\n"
            out += "\t\t)\n"
        return out

    def goal_any():
        # any ordered pair of objects may satisfy the task relation
        out = "\t\t(or\n"
        for i in range(n):
            for j in range(n):
                if i == j: continue
                for ts in task_syms:
                    out += f"\t\t\t(and {_bits_pred(ts,f'obj{i}',f'obj{j}')} (r{rel_dim} obj{i} obj{j}) )\n"
        out += "\t\t)\n"
        return out

    goal_str = goal_any() if goal_mode == "any" else goal_specific()
    prob = (f"(define (problem blocks-problem)\n\t(:domain blocks)\n"
            f"\t(:objects\n\t\t{' '.join(f'obj{i}' for i in range(n))} - object\n\t)\n"
            f"\t(:init\n\t\t(H0)\n\t\t(active-count-0)\n{obj_init()}{rel_init()}\t)\n"
            f"\t(:goal (and\n\t\t{usage}\n\t\t(not (active-count-collapse))\n{goal_str}\t))\n)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path,"w").write(prob)

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
        obj_t = [t for t in toks[1:] if re.match(r"^obj\d+$",t)]
        if obj_t:
            idx = int(obj_t[-1][3:])
            if idx < len(objects): placed.append(objects[idx])
    return placed

# ── simulation + result check (inline, serial — exactly like plan_task_clean) ─
def _find_existing_csv(path):
    """Mirrors plan_task_clean._find_existing_csv exactly."""
    for p in [path,
              path.replace(".csv","worker.csv"),
              os.path.join(os.path.dirname(path),
                           os.path.basename(path).replace(".csv","")+"worker.csv")]:
        if os.path.exists(p): return p
    return None

def run_sim_and_check(ordered, run_label, cfg):
    """
    Run get_experiment inline (serial), read CSV, check task pairs, collect rgb paths.
    Returns (collapsed, found_pairs, rgb_paths).
    Mirrors plan_task_clean.run_simulation_and_check exactly.
    """
    task = cfg["eval"]["task"]
    os.makedirs(cfg["_exp_dir"], exist_ok=True)

    full_label = f"{task}_{run_label}"
    try:
        get_experiment(ordered, cfg["_exp_dir"] + "/", full_label)
    except Exception as e:
        print(f"  [SIM ERROR] {e}")
        return True, [], []

    csv_path = os.path.join(cfg["_exp_dir"], f"{full_label}.csv")
    actual   = _find_existing_csv(csv_path)
    if actual is None:
        print(f"  [SIM ERROR] CSV not found: {csv_path}")
        return True, [], []

    try: df = load_csv(actual)
    except Exception as e:
        print(f"  [SIM ERROR] Cannot read CSV: {e}")
        return True, [], []

    # truncate at first collapse row (mirrors truncate_csv_at_collapse)
    keep, is_coll = [], False
    for _, row in df.iterrows():
        keep.append(row)
        if _row_collapsed(row): is_coll = True; break
    if is_coll:
        pd.DataFrame(keep).to_csv(actual, index=False)
        rgb = _collect_rgb(actual)
        return True, [], rgb

    found_pairs = []
    for _, row in df.iterrows():
        step = int(row.get("step",0))
        raw  = row.get("bounding_box_differences")
        if raw is None or (isinstance(raw,float) and pd.isna(raw)): continue
        try: diff = ast.literal_eval(str(raw))
        except: continue
        new_idx = step - 1
        for k, entry in diff.items():
            if not isinstance(entry,dict): continue
            if task not in str(entry.get("situation","")).lower(): continue
            ei = int(k)
            new_obj = ordered[new_idx] if new_idx < len(ordered) else None
            ex_obj  = ordered[ei]      if ei      < len(ordered) else None
            found_pairs.append({"new_obj": new_obj, "existing_obj": ex_obj})

    rgb = _collect_rgb(actual)
    return False, found_pairs, rgb

def _collect_rgb(csv_path):
    try: df = load_csv(csv_path)
    except: return []
    paths = []
    for _, row in df.sort_values("step").iterrows():
        p = str(row.get("rgb_image_path",""))
        if p and p != "nan" and os.path.exists(p): paths.append(p)
    return paths

def compute_success(expected, found):
    found_set = {(canonical(*fp["new_obj"]), canonical(*fp["existing_obj"]))
                 for fp in found if fp.get("new_obj") and fp.get("existing_obj")}
    matched = sum(1 for ep in expected
                  if ep.get("new_obj") and ep.get("existing_obj")
                  and (canonical(*ep["new_obj"]), canonical(*ep["existing_obj"])) in found_set)
    pct = 100.0 * matched / len(expected) if expected else 0.0
    return matched, pct

# ── image ─────────────────────────────────────────────────────────────────────
def save_image(cfg, idx, ordered, exp_pairs, found, pct, status, img_paths):
    os.makedirs(cfg["_img_dir"], exist_ok=True)
    IH, IW, CH = 240, 320, 130
    task = cfg["eval"]["task"]
    frames = []
    for p in img_paths:
        try: frames.append(Image.open(p).convert("RGB").resize((IW,IH),Image.LANCZOS))
        except: pass
    if not frames: frames = [Image.new("RGB",(IW,IH),(40,40,40))]
    canvas = Image.new("RGB",(IW*len(frames), IH+CH),(15,15,15))
    draw   = ImageDraw.Draw(canvas)
    for i,im in enumerate(frames):
        canvas.paste(im,(i*IW,0)); draw.text((i*IW+4,4),f"Step {i+1}",fill=(255,255,255))
    try:
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",13)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",11)
    except: fb = fs = ImageFont.load_default()
    color = {"SUCCESS":(80,200,120),"PARTIAL":(230,160,50),"FAIL":(220,80,60),
             "COLLAPSE":(180,80,220),"NO_PLAN":(255,165,0),
             "VERIFIED":(80,200,120),"UNVERIFIED":(180,180,60)}.get(status,(200,200,200))
    y = IH+6
    draw.text((8,y),f"[{idx}] {status}  {pct:.0f}%  task={task}  n={len(ordered)}",fill=color,font=fb); y+=18
    exp_str = ", ".join(f"{clean(e['new_obj'])}->{clean(e['existing_obj'])}" for e in exp_pairs)
    draw.text((8,y),f"expected: {exp_str[:120]}",fill=(200,200,200),font=fs); y+=15
    found_str = ", ".join(f"{clean(fp['new_obj'])}->{clean(fp['existing_obj'])}"
                          for fp in found if fp.get("new_obj") and fp.get("existing_obj")) if found else "none"
    draw.text((8,y),f"found:    {found_str[:120]}",fill=color,font=fs); y+=15
    draw.text((8,y),(" > ".join(clean(o) for o in ordered))[:120],fill=(150,150,150),font=fs)
    safe = re.sub(r"[^\w\-]","_",f"case_{idx:04d}_N{len(ordered)}_{status}_{int(pct):03d}")
    p = os.path.join(cfg["_img_dir"],f"{safe}.png"); canvas.save(p); return p

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config",default="configs/eval_task.yaml")
    args = parser.parse_args()

    cfg  = load_config(args.config); task = cfg["eval"]["task"]
    seed = int(cfg["eval"].get("seed",42))
    set_seed(seed); random.seed(seed)
    for d in [cfg["_pddl_dir"],cfg["_exp_dir"],cfg["_img_dir"]]: os.makedirs(d,exist_ok=True)
    if not os.path.exists(cfg["_domain"]): sys.exit(f"domain.pddl not found: {cfg['_domain']}")

    # model + symbols
    model, train_cfg = load_ckpt(cfg["model_name"],tag="best")
    cfg["_train_cfg"] = train_cfg; _load_data.init(train_cfg); set_seed(train_cfg["seed"])
    model.freeze(deterministic=True)
    rel_dim = train_cfg["model"]["symbol_size"]; obj_dim = train_cfg["model"]["obj_symbol_size"]
    collapse_syms = get_collapse_syms(model, cfg)
    task_syms     = get_task_syms(model, cfg)

    # build cases
    min_n = int(cfg["eval"].get("min_objects",2)); max_n = int(cfg["eval"].get("max_objects",4))
    df    = load_csv(cfg["data"]["dataset_csv"])
    print(f"\n--> Loaded {len(df)} rows. Building prefix cases N={min_n}..{max_n}...")
    all_cases, verify_index = build_cases(df, task, min_n, max_n)
    # attach pool to each case for verify_index lookup
    for c in all_cases:
        c["pool"] = tuple(sorted(canonical(str(r["object_type"]),str(r["object_size_or_path"]))
                                 for r in c["steps"]))
    print(f"  {len(all_cases)} cases  " + str({n:c for n,c in sorted(Counter(c["n_objects"] for c in all_cases).items())}))

    n_cap = int(cfg["eval"].get("n_baseline_scenarios",0))
    if n_cap > 0:
        all_cases = _sample_diverse(all_cases, n_cap)
        print(f"  After sampling {n_cap}/N: {len(all_cases)} cases")

    # object symbol cache
    unique_objs = list({canonical(str(r["object_type"]),str(r["object_size_or_path"]))
                        for c in all_cases for r in c["steps"]})
    syms      = get_obj_symbols(unique_objs, model, cfg)
    sym_cache = {o:s for o,s in zip(unique_objs,syms)}
    print(f"  {len(sym_cache)} unique object symbols cached\n")

    simulate  = bool(cfg["eval"].get("simulate", True))
    goal_mode = str(cfg["eval"].get("goal_mode", "specific")).lower()
    print(f"  goal_mode: {goal_mode}\n")
    results      = []
    t_plan_total = 0.0
    t_sim_total  = 0.0
    t_start      = time.time()

    for i, case in enumerate(all_cases):
        tid       = case["tower_id"]; n_objs = case["n_objects"]
        exp_pairs = case["exp_pairs"]
        objs      = [(str(r["object_type"]),str(r["object_size_or_path"]))
                     for r in sorted(case["steps"],key=lambda r:int(r["step"]))]
        obj_syms  = [sym_cache.get(canonical(*o)) for o in objs]
        if any(s is None for s in obj_syms):
            print(f"  [{i}] SKIP missing symbol"); continue

        prob_path = os.path.join(cfg["_pddl_dir"],f"p_{i:04d}_N{n_objs}_{tid}.pddl")
        write_problem(obj_syms, exp_pairs, task_syms, sym_cache,
                      rel_dim, obj_dim, prob_path, goal_mode)
        t0 = time.time()
        status, lines = run_planner(prob_path, cfg)
        t_plan = time.time() - t0
        t_plan_total += t_plan
        print(f"[{i+1}/{len(all_cases)}] tower={tid}  N={n_objs}  goal={goal_mode}  "
              f"planner={status}  ({t_plan:.1f}s)")

        if status != "SUCCESS":
            img = save_image(cfg,i,objs,exp_pairs,[],0.0,"NO_PLAN",[])
            results.append(dict(tower_id=tid,n_objects=n_objs,plan_status=status,
                                goal_mode=goal_mode,n_expected=len(exp_pairs),
                                n_matched=0,pct=0.0,status="NO_PLAN",
                                t_plan_sec=round(t_plan,2),t_sim_sec=0.0,
                                image_path=img)); continue

        ordered = plan_order(lines, objs)
        print(f"  order: {' > '.join(clean(o) for o in ordered)}")

        if not simulate:
            # verify against dataset
            t0        = time.time()
            order_key = tuple(canonical(*o) for o in ordered)
            known     = verify_index.get((case["pool"], n_objs), set())
            if goal_mode == "any":
                v_status = "VERIFIED" if order_key in known else "UNVERIFIED"
                v_note   = (f"order in data (task achieved)"
                            if v_status == "VERIFIED"
                            else f"order not in data; {len(known)} known seqs achieve task")
                n_matched, pct = (len(exp_pairs), 100.0) if v_status == "VERIFIED" else (0, 0.0)
            else:
                if order_key in known:
                    v_status = "VERIFIED"
                    v_note   = "exact order found in data with task achieved"
                    n_matched, pct = len(exp_pairs), 100.0
                else:
                    v_status = "UNVERIFIED"
                    v_note   = f"order not in data; {len(known)} known seqs achieve task for this pool"
                    n_matched, pct = 0, 0.0
            t_v = time.time() - t0
            print(f"  data-verify: {v_status}  ({v_note})  ({t_v*1000:.1f}ms)")
            img = save_image(cfg,i,ordered,exp_pairs,[],pct,v_status,[])
            print(f"  image -> {img}")
            results.append(dict(tower_id=tid,n_objects=n_objs,plan_status="SUCCESS",
                                goal_mode=goal_mode,n_expected=len(exp_pairs),
                                n_matched=n_matched,pct=pct,status=v_status,
                                verify_note=v_note,t_plan_sec=round(t_plan,2),
                                t_sim_sec=round(t_v,3),image_path=img))
            continue

        # run simulation inline — exactly like plan_task_clean.py
        run_label = f"{i:04d}_N{n_objs}_{tid}"
        t0        = time.time()
        is_coll, found, rgb = run_sim_and_check(ordered, run_label, cfg)
        t_sim = time.time() - t0
        t_sim_total += t_sim

        if is_coll:
            sim_status = "COLLAPSE"; n_matched = 0; pct = 0.0
        else:
            n_matched, pct = compute_success(exp_pairs, found)
            sim_status = "SUCCESS" if pct==100 else "PARTIAL" if pct>0 else "FAIL"

        print(f"  sim: {sim_status}  {n_matched}/{len(exp_pairs)}  ({pct:.0f}%)  ({t_sim:.1f}s)")
        img = save_image(cfg,i,ordered,exp_pairs,found,pct,sim_status,rgb)
        print(f"  image -> {img}")
        results.append(dict(tower_id=tid,n_objects=n_objs,plan_status="SUCCESS",
                            goal_mode=goal_mode,n_expected=len(exp_pairs),
                            n_matched=n_matched,pct=pct,status=sim_status,
                            t_plan_sec=round(t_plan,2),t_sim_sec=round(t_sim,2),
                            image_path=img))

    t_total = time.time() - t_start

    df_r = pd.DataFrame(results)
    df_r.to_csv(cfg["_results"],index=False)

    print(f"\n{'='*50}\nSUMMARY  task={task}  goal={goal_mode}  simulate={simulate}\n{'='*50}")
    for n in range(min_n,max_n+1):
        sub = df_r[df_r["n_objects"]==n]
        if sub.empty: continue
        ran = sub[sub["plan_status"]=="SUCCESS"]
        if simulate:
            avg = f"{ran['pct'].mean():.1f}%" if len(ran) else "—"
            print(f"  N={n}: total={len(sub)}  "
                  f"success={(sub['status']=='SUCCESS').sum()}  "
                  f"partial={(sub['status']=='PARTIAL').sum()}  "
                  f"fail={(sub['status']=='FAIL').sum()}  "
                  f"no_plan={(sub['status']=='NO_PLAN').sum()}  "
                  f"collapse={(sub['status']=='COLLAPSE').sum()}  avg_pct={avg}")
        else:
            print(f"  N={n}: total={len(sub)}  "
                  f"verified={(sub['status']=='VERIFIED').sum()}  "
                  f"unverified={(sub['status']=='UNVERIFIED').sum()}  "
                  f"no_plan={(sub['status']=='NO_PLAN').sum()}")

    n_cases = len(results)
    print(f"\n── Timing ──────────────────────────────────────")
    print(f"  total            : {t_total:.1f}s")
    print(f"  planning         : {t_plan_total:.1f}s  (avg {t_plan_total/max(n_cases,1):.2f}s/case)")
    if simulate:
        print(f"  simulation       : {t_sim_total:.1f}s  (avg {t_sim_total/max(n_cases,1):.2f}s/case)")
    print(f"  results -> {cfg['_results']}\n  images  -> {cfg['_img_dir']}")

if __name__ == "__main__":
    main()