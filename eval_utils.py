"""
eval_utils.py — shared helpers for plan_height_fast.py and plan_task_fast.py.
"""
import ast, os, re, signal, subprocess, sys
from collections import defaultdict

import numpy as np, pandas as pd, torch, yaml
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import load_ckpt
from symbol_semantics_fast import get_semantic_symbols
from data_collection_direct.exp import get_experiment
import load_data as _load_data
from load_data import set_seed

# ── naming ────────────────────────────────────────────────────────────────────

def canonical(t, s):
    try: float(s); return (str(t), str(s))
    except ValueError: return (str(t), os.path.basename(str(s)).replace(".urdf", ""))

def clean(o):
    c = canonical(*o); return f"{c[0]}_{c[1]}"

# ── CSV / IO ──────────────────────────────────────────────────────────────────

def load_csv(p):
    for enc in ["utf-8", "cp1254", "latin-1"]:
        try: return pd.read_csv(p, encoding=enc)
        except Exception: pass
    sys.exit(f"Cannot read {p}")

def find_sim_csv(path):
    stem = path.replace(".csv", "")
    for p in [path, stem + "worker.csv",
              os.path.join(os.path.dirname(path), os.path.basename(stem) + "worker.csv")]:
        if os.path.exists(p): return p
    return None

def row_collapsed(row):
    v = row.get("collapse", 0)
    try: return float(v) >= 0.5
    except: return str(v).strip().lower() in ("true", "1", "yes")

def collect_rgb(csv_path):
    try: df = load_csv(csv_path)
    except: return []
    return [p for p in (str(r.get("rgb_image_path", ""))
                        for _, r in df.sort_values("step").iterrows())
            if p and p != "nan" and os.path.exists(p)]

# ── model / symbols ───────────────────────────────────────────────────────────

def load_model(cfg):
    model, train_cfg = load_ckpt(cfg["model_name"], tag="best")
    cfg["_train_cfg"] = train_cfg
    _load_data.init(train_cfg); set_seed(train_cfg["seed"])
    model.freeze(deterministic=True)
    return model, train_cfg

def depth_img(otype, opath, depth_dir):
    base = f"{otype}_{canonical(otype, opath)[1]}"
    up = os.path.join(depth_dir, f"upper_{base}_new.npy")
    lo = os.path.join(depth_dir, f"lower_{base}_new.npy")
    if os.path.exists(up) and os.path.exists(lo):
        return torch.cat([torch.from_numpy(np.load(up)).unsqueeze(0).float(),
                          torch.from_numpy(np.load(lo)).unsqueeze(0).float()], 0)
    print(f"  [WARN] depth images missing for {base}")
    return torch.zeros(2, 64, 64)

def get_obj_symbols(objects, model, depth_dir):
    dev = next(model.parameters()).device
    model.eval(); model.gs_obj_layer.hard = model.gs_obj_layer.deterministic = True
    results = []
    with torch.no_grad():
        for ot, op in objects:
            img   = depth_img(ot, op, depth_dir)
            batch = next(iter(DataLoader([Data(x=img.unsqueeze(0),
                              edge_index=torch.empty((2,0), dtype=torch.long),
                              edge_attr =torch.empty((0,5), dtype=torch.float))], batch_size=1)))
            feats = model.image_encoder(batch.x.float().to(dev))
            sim   = F.normalize(feats, dim=-1) @ F.normalize(model.cluster_centroids, dim=-1).T
            results.append(model.cluster_codes[sim.argmax(dim=-1)].squeeze(0).cpu().int().tolist())
    return results

def get_collapse_syms(model, train_cfg):
    dev = next(model.parameters()).device
    syms, _, _ = get_semantic_symbols(model, sym_size=train_cfg["model"]["symbol_size"],
                                      out_dim=4, collapse_threshold=0.5, device=dev)
    return [list(s) for s in syms]

# ── PDDL helpers ──────────────────────────────────────────────────────────────

def bits_pred(bits, v1, v2):
    return "(and " + " ".join(f"(r{i} {v1} {v2})" if b else f"(not (r{i} {v1} {v2}))"
                              for i, b in enumerate(bits)) + ")"

def obj_init_pddl(obj_syms, obj_dim):
    lines = []
    for i, bits in enumerate(obj_syms):
        z = " ".join(f"(z{j} obj{i})" if v else f"(not (z{j} obj{i}))" for j, v in enumerate(bits))
        lines.append(f"\t\t(top-0 obj{i})\n\t\t{z} (not (z{obj_dim} obj{i}))")
    return "\n".join(lines) + "\n"

def rel_init_pddl(n, rel_dim):
    return "\n".join(
        f"\t\t" + " ".join(f"(not (r{k} obj{i} obj{j}))" for k in range(rel_dim + 1))
        for i in range(n) for j in range(n) if i != j
    ) + "\n"

# ── planner ───────────────────────────────────────────────────────────────────

def run_planner(problem_path, cfg):
    pf, t = cfg["_plan_file"], int(cfg["planner"].get("time_limit_sec", 10))
    if os.path.exists(pf): os.remove(pf)
    cmd = [sys.executable, cfg["planner"]["fast_downward"], "--plan-file", pf,
           cfg["_domain"], problem_path, "--search", cfg["planner"]["search"]]
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        proc.wait(timeout=t)
    except subprocess.TimeoutExpired:
        try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError: pass
        return "TIMEOUT", []
    except Exception as e:
        return f"ERROR_{e}", []
    if os.path.exists(pf):
        return "SUCCESS", [l.strip() for l in open(pf) if l.strip() and not l.startswith(";")]
    return "NO_PLAN", []

def plan_order(lines, objects):
    placed = []
    for line in lines:
        toks = line.lstrip("(").rstrip(")").split()
        if not toks or not toks[0].startswith("a_"): continue
        obj_toks = [t for t in toks[1:] if re.match(r"^obj\d+$", t)]
        if obj_toks:
            idx = int(obj_toks[-1][3:])
            if idx < len(objects): placed.append(objects[idx])
    return placed

# ── simulation ────────────────────────────────────────────────────────────────

def run_sim(ordered, run_label, exp_dir):
    """Run get_experiment; return (df, hard_error, rgb_paths)."""
    os.makedirs(exp_dir, exist_ok=True)
    try: get_experiment(ordered, exp_dir + "/", run_label)
    except Exception as e: print(f"  [SIM ERROR] {e}"); return None, True, []
    csv_path = os.path.join(exp_dir, f"{run_label}.csv")
    actual   = find_sim_csv(csv_path)
    if actual is None: print(f"  [SIM ERROR] CSV not found: {csv_path}"); return None, True, []
    try: df = load_csv(actual)
    except Exception as e: print(f"  [SIM ERROR] Cannot read CSV: {e}"); return None, True, []
    return df, False, collect_rgb(actual)

def parse_bbox_df(df):
    """Return (step_bboxes, is_collapsed) from a simulation CSV DataFrame."""
    step_bboxes, is_coll = {}, False
    for _, row in df.iterrows():
        if row_collapsed(row): is_coll = True
        step = int(row.get("step", 0))
        raw  = row.get("bbox")
        if raw is None or (isinstance(raw, float) and pd.isna(raw)): continue
        try:
            bbox = ast.literal_eval(str(raw)) if not isinstance(raw, dict) else raw
            if bbox: step_bboxes[step] = {int(k): v for k, v in bbox.items()}
        except Exception: pass
    return step_bboxes, is_coll

# ── image saving ──────────────────────────────────────────────────────────────

STATUS_COLORS = {
    "SUCCESS": (80,200,120), "PARTIAL": (230,160,50), "FAIL": (220,80,60),
    "COLLAPSE": (180,80,220), "NO_PLAN": (255,165,0),
    "VERIFIED": (80,200,120), "UNVERIFIED": (180,180,60), "WRONG_H": (220,80,60),
}

def _fonts():
    try:
        return (ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13),
                ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11))
    except OSError:
        d = ImageFont.load_default(); return d, d

def _frame_strip(img_paths, iw=320, ih=256):
    frames = []
    for p in img_paths:
        try: frames.append(Image.open(p).convert("RGB").resize((iw, ih), Image.LANCZOS))
        except: pass
    if not frames: frames = [Image.new("RGB", (iw, ih), (40,40,40))]
    strip = Image.new("RGB", (iw * len(frames), ih), (15,15,15))
    fb, _ = _fonts(); draw = ImageDraw.Draw(strip)
    for i, f in enumerate(frames):
        strip.paste(f, (i*iw, 0)); draw.text((i*iw+4, 4), f"step {i+1}", (255,255,255), font=fb)
    return strip

def _save_image(img_dir, fname, img_paths, ih, ch, annotation_fn):
    os.makedirs(img_dir, exist_ok=True)
    strip  = _frame_strip(img_paths, ih=ih)
    canvas = Image.new("RGB", (strip.width, ih + ch), (15,15,15))
    canvas.paste(strip, (0, 0))
    draw = ImageDraw.Draw(canvas)
    annotation_fn(draw, _fonts(), ih)
    path = os.path.join(img_dir, fname + ".png")
    canvas.save(path); return path

def save_image_height(img_dir, label, ordered, target_h, sim_h, status, img_paths):
    color = STATUS_COLORS.get(status, (200,200,200))
    order_str = (" > ".join(clean(o) for o in ordered))[:120]
    def annotate(draw, fonts, ih):
        fb, fs = fonts
        draw.text((8, ih+4),  f"{label}  {status}  target=H{target_h}  sim=H{sim_h}", color, font=fb)
        draw.text((8, ih+22), order_str, (150,150,150), font=fs)
    fname = re.sub(r"[^\w\-]", "_", f"{label}_{status}_H{target_h}")
    return _save_image(img_dir, fname, img_paths, ih=256, ch=100, annotation_fn=annotate)

def save_image_task(img_dir, idx, ordered, exp_pairs, found, pct, status, img_paths, task):
    color     = STATUS_COLORS.get(status, (200,200,200))
    exp_str   = ", ".join(f"{clean(e['new_obj'])}->{clean(e['existing_obj'])}" for e in exp_pairs)
    found_str = (", ".join(f"{clean(fp['new_obj'])}->{clean(fp['existing_obj'])}"
                           for fp in found if fp.get("new_obj") and fp.get("existing_obj"))
                 if found else "none")
    order_str = (" > ".join(clean(o) for o in ordered))[:120]
    def annotate(draw, fonts, ih):
        fb, fs = fonts; y = ih + 6
        draw.text((8,y), f"[{idx}] {status}  {pct:.0f}%  task={task}  n={len(ordered)}", color, font=fb); y+=18
        draw.text((8,y), f"expected: {exp_str[:120]}",   (200,200,200), font=fs); y+=15
        draw.text((8,y), f"found:    {found_str[:120]}", color,         font=fs); y+=15
        draw.text((8,y), order_str, (150,150,150), font=fs)
    fname = re.sub(r"[^\w\-]", "_", f"case_{idx:04d}_N{len(ordered)}_{status}_{int(pct):03d}")
    return _save_image(img_dir, fname, img_paths, ih=240, ch=130, annotation_fn=annotate)

# ── summary ───────────────────────────────────────────────────────────────────

def _pct(num, den): return f"{100*num/den:.1f}%" if den else "—"

def print_summary(df_r, title, min_n, max_n, simulate, t_total, t_plan, t_sim, cfg, mode="height"):
    W = 108 if mode == "task" else 100
    print(f"\n{'='*W}\n  SUMMARY  {title}\n{'='*W}")

    # build per-row stats using groupby for brevity
    cols = ["status", "plan_status", "t_plan_sec"] + (["t_sim_sec", "pct"] if simulate else [])
    grp  = df_r.groupby("n_objects")

    tots = defaultdict(float)
    for n, sub in grp:
        if n < min_n or n > max_n: continue
        _print_summary_row(sub, n, simulate, mode, tots)

    print("-" * W)
    T = int(tots["total"])
    total_sub = df_r[(df_r["n_objects"] >= min_n) & (df_r["n_objects"] <= max_n)]
    _print_summary_row(total_sub, "TOTAL", simulate, mode, tots, T=T)

    print(f"\n── Timing {'─'*40}")
    print(f"  total wall time  : {t_total:.1f}s")
    print(f"  planning total   : {t_plan:.1f}s  (avg {t_plan/max(T,1):.2f}s/case)")
    if simulate: print(f"  simulation total : {t_sim:.1f}s  (avg {t_sim/max(T,1):.2f}s/case)")
    print(f"  results -> {cfg['_results']}\n  images  -> {cfg['_img_dir']}")
    print(f"{'='*W}\n")

def _print_summary_row(sub, label, simulate, mode, tots, T=None):
    """Print one data row and accumulate into tots. If T given, print TOTAL row from tots."""
    if T is not None:  # TOTAL row
        tot = T
        if simulate and mode == "height":
            t_pa = tots["t_plan"]/max(T,1); t_sa = tots["t_sim"]/max(T,1)
            print(f"  {'TOTAL':<8}  {T:>5}  {int(tots['no_plan']):>7}  {_pct(tots['no_plan'],T):>8}  "
                  f"{int(tots['success']):>7}  {_pct(tots['success'],T):>8}  "
                  f"{int(tots['fail']):>4}  {_pct(tots['fail'],T):>5}  "
                  f"{int(tots['collapse']):>8}  {_pct(tots['collapse'],T):>9}  "
                  f"{t_pa:>10.2f}s  {t_sa:>8.2f}s")
        elif simulate and mode == "task":
            t_pa = tots["t_plan"]/max(T,1); t_sa = tots["t_sim"]/max(T,1)
            avg  = f"{tots['pct_sum']/tots['pct_n']:.1f}%" if tots["pct_n"] else "—"
            print(f"  {'TOTAL':<5}  {T:>5}  {int(tots['no_plan']):>7}  {_pct(tots['no_plan'],T):>8}  "
                  f"{int(tots['success']):>7}  {_pct(tots['success'],T):>8}  "
                  f"{int(tots['partial']):>7}  {_pct(tots['partial'],T):>8}  "
                  f"{int(tots['fail']):>4}  {_pct(tots['fail'],T):>5}  "
                  f"{int(tots['collapse']):>8}  {_pct(tots['collapse'],T):>9}  "
                  f"{avg:>7}  {t_pa:>10.2f}s  {t_sa:>8.2f}s")
        else:
            t_pa = tots["t_plan"]/max(T,1)
            print(f"  {'TOTAL':<5}  {T:>5}  {int(tots['no_plan']):>7}  {_pct(tots['no_plan'],T):>8}  "
                  f"{int(tots.get('planned',0)):>7}  {_pct(tots.get('planned',0),T):>8}  "
                  f"{int(tots['timeout']):>7}  {_pct(tots['timeout'],T):>8}  {t_pa:>10.2f}s")
        return

    tot     = len(sub)
    no_plan = (sub["status"]   == "NO_PLAN").sum()
    timeout = (sub["plan_status"] == "TIMEOUT").sum()
    t_pa    = sub["t_plan_sec"].mean()
    tots["total"] += tot; tots["no_plan"] += no_plan
    tots["timeout"] += timeout; tots["t_plan"] += sub["t_plan_sec"].sum()

    if simulate:
        success  = (sub["status"] == "SUCCESS").sum()
        fail     = (sub["status"] == "FAIL").sum()
        collapse = (sub["status"] == "COLLAPSE").sum()
        t_sa     = sub["t_sim_sec"].mean()
        tots["success"]+=success; tots["fail"]+=fail
        tots["collapse"]+=collapse; tots["t_sim"]+=sub["t_sim_sec"].sum()
        if mode == "height":
            print(f"  {label!s:<8}  {tot:>5}  {no_plan:>7}  {_pct(no_plan,tot):>8}  "
                  f"{success:>7}  {_pct(success,tot):>8}  {fail:>4}  {_pct(fail,tot):>5}  "
                  f"{collapse:>8}  {_pct(collapse,tot):>9}  {t_pa:>10.2f}s  {t_sa:>8.2f}s")
        else:
            partial = (sub["status"] == "PARTIAL").sum()
            ran     = sub[sub["plan_status"] == "SUCCESS"]
            avg_pct = f"{ran['pct'].mean():.1f}%" if len(ran) else "—"
            tots["partial"]+=partial
            if len(ran): tots["pct_sum"]+=ran["pct"].sum(); tots["pct_n"]+=len(ran)
            print(f"  {label!s:<5}  {tot:>5}  {no_plan:>7}  {_pct(no_plan,tot):>8}  "
                  f"{success:>7}  {_pct(success,tot):>8}  {partial:>7}  {_pct(partial,tot):>8}  "
                  f"{fail:>4}  {_pct(fail,tot):>5}  {collapse:>8}  {_pct(collapse,tot):>9}  "
                  f"{avg_pct:>7}  {t_pa:>10.2f}s  {t_sa:>8.2f}s")
    else:
        if mode == "height":
            planned = (sub["plan_status"] == "SUCCESS").sum()
            tots["planned"] += planned
            print(f"  {label!s:<8}  {tot:>5}  {no_plan:>7}  {_pct(no_plan,tot):>8}  "
                  f"{planned:>7}  {_pct(planned,tot):>8}  "
                  f"{timeout:>7}  {_pct(timeout,tot):>8}  {t_pa:>10.2f}s")
        else:
            verified   = (sub["status"] == "VERIFIED").sum()
            unverified = (sub["status"] == "UNVERIFIED").sum()
            tots["verified"]+=verified; tots["unverified"]+=unverified
            print(f"  {label!s:<5}  {tot:>5}  {no_plan:>7}  {_pct(no_plan,tot):>8}  "
                  f"{verified:>8}  {_pct(verified,tot):>9}  "
                  f"{unverified:>10}  {_pct(unverified,tot):>8}  "
                  f"{timeout:>7}  {_pct(timeout,tot):>8}  {t_pa:>10.2f}s")

# ── case sampling ─────────────────────────────────────────────────────────────

def sample_diverse(all_cases, n_per_n):
    """Per N: pick up to n_per_n cases, prioritising pool diversity."""
    by_n = defaultdict(list)
    for c in all_cases: by_n[c["n_objects"]].append(c)
    result = []
    for n in sorted(by_n):
        by_pool = defaultdict(list)
        for c in by_n[n]: by_pool[c["pool"]].append(c)
        pools    = list(by_pool.keys())
        selected = [by_pool[p].pop() for p in pools]
        idx = 0
        while len(selected) < n_per_n and idx < len(pools) * (n_per_n + 1):
            if by_pool[pools[idx % len(pools)]]:
                selected.append(by_pool[pools[idx % len(pools)]].pop())
            idx += 1
        result.extend(selected[:n_per_n])
    return result