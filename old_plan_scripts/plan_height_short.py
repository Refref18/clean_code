"""
plan_height.py
==============
For every object group where min_levels == target_min AND max_levels == target_max:
  - Run PDDL planner for both heights
  - Simulate the resulting object order
  - Record success/failure

Usage:
    python plan_height.py -c configs/eval.yaml
"""

import argparse, ast, json, os, re, shutil, subprocess, sys
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


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    name          = cfg["model_name"]
    save_root     = os.path.join("save", name)
    eval_dir      = os.path.join(save_root, "eval")
    cfg["_domain"]    = os.path.join(save_root, "domain.pddl")
    cfg["_pddl_dir"]  = os.path.join(save_root, "PDDL_FILES")
    cfg["_plan_file"] = os.path.join(save_root, "PDDL_FILES", "sas_plan")
    cfg["_eval_dir"]  = eval_dir
    cfg["_exp_dir"]   = os.path.join(eval_dir, "sim_experiments")
    cfg["_img_dir"]   = os.path.join(eval_dir, "images")
    cfg["_results"]   = os.path.join(eval_dir, "results.csv")
    return cfg


# ── CSV helper ────────────────────────────────────────────────────────────────

def load_csv(path):
    for enc in ["utf-8", "cp1254", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    sys.exit(f"Cannot read {path}")


# ── Object symbols ────────────────────────────────────────────────────────────

KNOWN_SIZES = [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13,
               0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]


def _clean_size(obj_size_or_path):
    try:
        val = float(obj_size_or_path)
        return str(min(KNOWN_SIZES, key=lambda x: abs(x - val)))
    except (ValueError, TypeError):
        return str(obj_size_or_path).split('/')[-1].replace('.urdf', '')


def load_depth_image(obj_type, obj_size_or_path, depth_dir, device):
    clean   = _clean_size(obj_size_or_path)
    base    = f"{obj_type}_{clean}"
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
    Compute object symbols via single-item DataLoader per object so that
    BatchNorm uses running stats (eval mode) with the same batch structure
    as during training — without running the full preprocess pipeline that
    opens PyBullet clients and corrupts subsequent simulations.
    """
    import torch.nn.functional as F
    depth_dir = cfg["data"]["depth_image_dir"]
    device    = next(model.parameters()).device

    model.eval()
    model.gs_obj_layer.hard          = True
    model.gs_obj_layer.deterministic = True

    results = []
    with torch.no_grad():
        for otype, opath in ordered_objects:
            img = load_depth_image(otype, opath, depth_dir, torch.device("cpu"))
            # Wrap in a single-item DataLoader so BatchNorm sees the same
            # batch structure as during training (each object as one graph node)
            single = Data(x=img.unsqueeze(0),
                          edge_index=torch.empty((2, 0), dtype=torch.long),
                          edge_attr=torch.empty((0, 5), dtype=torch.float))
            loader = DataLoader([single], batch_size=1, shuffle=False, num_workers=0)
            batch  = next(iter(loader))
            x      = batch.x.float().to(device)
            feats  = model.image_encoder(x)
            sim    = F.normalize(feats, dim=-1) @ F.normalize(model.cluster_centroids, dim=-1).T
            bits   = model.cluster_codes[sim.argmax(dim=-1)]
            results.append(bits.squeeze(0).cpu().int().tolist())
    return results


# ── Height analysis ───────────────────────────────────────────────────────────

def _parse(val):
    try:
        return val if isinstance(val, (list, dict)) else ast.literal_eval(str(val))
    except Exception:
        return None

def _obj_key(row):
    fn = os.path.basename(str(row.get("object_size_or_path",""))).replace(".urdf","")
    parts = fn.split("_")
    stem = "_".join(parts[:3]) if len(parts) >= 3 else parts[0]
    return f"{row.get('object_type','unknown')}_{stem}"

def _levels(step_bboxes, thr):
    levels = 0
    for s in sorted(step_bboxes):
        bb = step_bboxes[s]
        if not bb: continue
        idx = s - 1 if s - 1 in bb else max(bb)
        zmax = bb[idx]["max"][2]
        if levels == 0:
            levels = 1
        elif [bb[i]["max"][2] for i in bb if i != idx] and \
             zmax > max(bb[i]["max"][2] for i in bb if i != idx) + thr:
            levels += 1
    return levels

def compute_height_analysis(csv_path, z_thr):
    df = load_csv(csv_path)
    tower_objs, tower_bboxes, tower_collapsed = defaultdict(dict), defaultdict(dict), {}
    for _, row in df.iterrows():
        tid, step = row["id"], int(row["step"])
        tower_objs[tid][step] = _obj_key(row)
        bb = _parse(row.get("bbox"))
        if bb: tower_bboxes[tid][step] = bb
        try:
            if float(row.get("collapse", 0)) >= 0.5: tower_collapsed[tid] = True
        except (TypeError, ValueError): pass

    records = []
    for tid, steps in tower_objs.items():
        objs      = [steps[s] for s in sorted(steps)]
        collapsed = tower_collapsed.get(tid, False)
        bboxes    = tower_bboxes.get(tid, {})
        records.append({
            "tower_id":     tid,
            "n_objects":    len(objs),
            "object_group": str(tuple(sorted(objs))),
            "object_order": str(tuple(objs)),
            "collapsed":    collapsed,
            "levels":       _levels(bboxes, z_thr) if not collapsed else None,
        })

    df_det  = pd.DataFrame(records)
    df_val  = df_det[~df_det["collapsed"]].copy()
    if df_val.empty:
        return df_det, pd.DataFrame()

    agg = df_val.groupby("object_group").agg(
        n_objects  =("n_objects", "first"),
        min_levels =("levels",    "min"),
        max_levels =("levels",    "max"),
    ).reset_index()

    # attach one example order for min/max level
    for label, col in [("min","min_level_order"), ("max","max_level_order")]:
        idx  = df_val.groupby("object_group")["levels"].idxmin() if label=="min" \
               else df_val.groupby("object_group")["levels"].idxmax()
        rows = df_val.loc[idx, ["object_group","object_order"]].rename(
               columns={"object_order": col})
        agg  = agg.merge(rows, on="object_group", how="left")

    print(f"  [height_analysis] {len(df_det)} rows | {len(agg)} groups")
    return df_det, agg


# ── Dataset helpers ───────────────────────────────────────────────────────────

def build_pair_to_original(df):
    lookup = {}
    for _, row in df.iterrows():
        otype = str(row.get("object_type",""))
        raw   = str(row.get("object_size_or_path",""))
        parts = os.path.basename(raw).replace(".urdf","").split("_")
        key   = f"{otype}_{'_'.join(parts[:3]) if len(parts)>=3 else parts[0]}"
        if key not in lookup:
            lookup[key] = (otype, row.get("object_size_or_path"))
    return lookup

def group_to_objects(group_str, pair_to_original):
    try:
        keys = ast.literal_eval(group_str)
    except Exception:
        return None, None
    objs, order = [], []
    for k in keys:
        orig = pair_to_original.get(k)
        if orig is None: return None, None
        objs.append(orig); order.append(k)
    return objs, order

def known_sequences(df_det, group_label, height):
    sub = df_det[(df_det["object_group"]==group_label) &
                 (df_det["levels"]==height) & (~df_det["collapsed"])]
    out = set()
    for _, row in sub.iterrows():
        try: out.add(tuple(ast.literal_eval(str(row["object_order"]))))
        except Exception: pass
    return out


# ── PDDL construction ─────────────────────────────────────────────────────────

def _obj_syms_to_pddl(obj_symbols, obj_dim, ind="\t\t"):
    out = ""
    for i, sym in enumerate(obj_symbols):
        out += f"{ind}(top-0 obj{i})\n{ind}"
        out += " ".join(f"(z{j} obj{i})" if v else f"(not_z{j} obj{i})" for j,v in enumerate(sym))
        out += f" (not_z{obj_dim} obj{i}) \n"
    return out

def _rel_init_to_pddl(n, rel_dim, ind="\t\t"):
    out = ""
    for i in range(n):
        for j in range(n):
            if i==j: continue
            out += ind + " ".join(f"(not_r{k} obj{i} obj{j})" for k in range(rel_dim+1)) + "\n"
    return out

def _spec_pairs(all_syms, spec_syms):
    idxs = [[i for i,v in enumerate(all_syms) if v==t] for t in spec_syms]
    return [tuple(f"obj{i}" for i in combo) for combo in product(*idxs)]

def _goal_restriction(req_list, all_syms, rel_dim, ind="\t\t"):
    out = ""
    for case in req_list:
        for rsym in case["pos_rel_symbols"]:
            out += ind + "(or\n"
            for n1,n2 in _spec_pairs(all_syms, case["obj_symbols"]):
                out += ind+"\t(and "
                out += " ".join(f"(r{k} {n1} {n2})" if v else f"(not_r{k} {n1} {n2})" for k,v in enumerate(rsym))
                out += f" (r{rel_dim} {n1} {n2}) )\n"
            out += ind + ")\n"
    return out

def write_problem_pddl(obj_syms, collapse_syms, req_list, rel_dim, obj_dim, height, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    n = len(obj_syms)
    txt  = "(define (problem blocks-problem)\n\t(:domain blocks)\n"
    txt += f"\t(:objects\n\t\t{' '.join(f'obj{i}' for i in range(n))} - object\n\t)\n"
    txt += "\t(:init\n\t\t(H0)\n\t\t(active-count-0)\n"
    txt += _obj_syms_to_pddl(obj_syms, obj_dim)
    txt += _rel_init_to_pddl(n, rel_dim)
    txt += "\t)\n\t(:goal (and\n"
    txt += f"\t\t(H{height})\n\t\t(all-used)\n\t\t(not (active-count-collapse))\n"
    txt += _goal_restriction(req_list, obj_syms, rel_dim)
    txt += "\t))\n)"
    path = os.path.join(save_dir, "problem.pddl")
    with open(path,"w") as f: f.write(txt)
    return path


# ── Planner ───────────────────────────────────────────────────────────────────

def run_planner(cfg):
    plan_file = cfg["_plan_file"]
    if os.path.exists(plan_file): os.remove(plan_file)
    r = subprocess.run(
        [sys.executable, cfg["planner"]["fast_downward"],
         "--plan-file", plan_file,
         cfg["_domain"], os.path.join(cfg["_pddl_dir"],"problem.pddl"),
         "--search", cfg["planner"]["search"]],
        capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.returncode == 0 and os.path.exists(plan_file):
        with open(plan_file) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith(";")]
        return "SUCCESS", lines, out
    return ("NO_PLAN", [], out) if (r.returncode==12 or "NO PLAN FOUND" in out) \
           else (f"ERROR_{r.returncode}", [], out)

def extract_order(plan_lines, pair_order):
    placed = []
    for line in plan_lines:
        tokens = line.strip().lstrip("(").rstrip(")").split()
        if not tokens or not tokens[0].startswith("a_"): continue
        toks = [t for t in tokens[1:] if re.match(r"^obj\d+$",t)]
        if toks:
            idx = int(toks[-1].replace("obj",""))
            if idx < len(pair_order): placed.append(pair_order[idx])
    return tuple(placed) if placed else None

def save_pddl_copy(cfg, group_idx, height, status):
    dst = os.path.join(cfg["_pddl_dir"], "successful_plans" if status=="SUCCESS" else "no_plan_files")
    os.makedirs(dst, exist_ok=True)
    label = f"group_{group_idx}_H{height}_{status}"
    for src, ext in [(os.path.join(cfg["_pddl_dir"],"problem.pddl"),".pddl"),
                     (cfg["_plan_file"],".sas_plan")]:
        if os.path.exists(src): shutil.copy(src, os.path.join(dst, label+ext))


# ── Simulation ────────────────────────────────────────────────────────────────

def _parse_bbox(val):
    try:
        return val if isinstance(val,dict) else ast.literal_eval(str(val))
    except Exception: return None

def read_sim_levels(csv_path, z_thr):
    if not os.path.exists(csv_path):
        fallback = csv_path.replace(".csv","worker.csv")
        if os.path.exists(fallback): csv_path = fallback
        else: return None, True
    try: df = load_csv(csv_path)
    except Exception: return None, True
    bboxes, collapsed = {}, False
    for _, row in df.iterrows():
        step = int(row.get("step",0))
        cv   = row.get("collapse",0)
        try:
            if float(cv) >= 0.5: collapsed = True
        except (TypeError,ValueError):
            if str(cv).strip().lower() in ("true","1","yes"): collapsed = True
        bb = _parse_bbox(row.get("bbox"))
        if bb: bboxes[step] = {int(k):v for k,v in bb.items()}
    if collapsed: return None, True
    return _levels(bboxes, z_thr), False

def run_simulation(plan_order, pair_to_original, cfg, group_idx, height):
    os.makedirs(cfg["_exp_dir"], exist_ok=True)
    label   = f"group_{group_idx}_H{height}"
    z_thr   = cfg["eval"]["z_level_threshold"]
    objects = [pair_to_original[k] for k in plan_order]
    print(f"  [SIM] " + " > ".join(f"{o[0]}|{os.path.basename(str(o[1]))}" for o in objects))

    import pybullet as _pb
    for _cid in range(20):
        try: _pb.disconnect(_cid)
        except Exception: pass

    try:
        get_experiment(objects, cfg["_exp_dir"]+"/", label)
    except Exception as e:
        print(f"  [SIM] Exception: {e}")
        return None, False, True, None

    csv_path = os.path.join(cfg["_exp_dir"], f"{label}.csv")
    sim_lvl, collapsed = read_sim_levels(csv_path, z_thr)
    if collapsed:
        print(f"  [SIM] Collapsed")
        return None, False, True, csv_path
    if sim_lvl is None:
        return None, False, False, csv_path
    ok = sim_lvl == height
    print(f"  [SIM] sim={sim_lvl} target={height} {'✓' if ok else '✗'}")
    return sim_lvl, ok, False, csv_path


# ── Images ────────────────────────────────────────────────────────────────────

_IH=256; _CH=120; _BG=(15,15,15); _FG=(255,255,255)

def get_rgb_paths(csv_path):
    for cand in [csv_path, csv_path.replace(".csv","worker.csv"),
                 os.path.join(os.path.dirname(csv_path),
                              os.path.basename(csv_path).replace(".csv","")+"worker.csv")]:
        if os.path.exists(cand):
            try:
                df = load_csv(cand)
                paths = [str(r.get("rgb_image_path","")) for _,r in df.sort_values("step").iterrows()]
                return [p for p in paths if p and p!="nan" and os.path.exists(p)]
            except Exception: continue
    return []

def save_image(cfg, group_label, group_idx, height, plan_str, matched,
               sim_lvl, sim_ok, collapsed, status, rgb_paths, os_swap=None):
    os.makedirs(cfg["_img_dir"], exist_ok=True)
    tw = int(_IH*4/3)
    imgs = []
    for p in rgb_paths:
        try:
            im = Image.open(p).convert("RGB")
            im = im.resize((max(1,int(im.width*_IH/im.height)), _IH), Image.LANCZOS)
            imgs.append(im)
        except Exception: pass
    if not imgs: imgs = [Image.new("RGB",(tw,_IH),(40,40,40))]
    cw     = max(max(i.width for i in imgs), tw)
    canvas = Image.new("RGB",(cw*len(imgs),_IH+_CH),_BG)
    draw   = ImageDraw.Draw(canvas)
    for k,im in enumerate(imgs):
        canvas.paste(im,(k*cw+(cw-im.width)//2,0))
        draw.text((k*cw+5,5),f"step {k+1}",fill=_FG)
    try:
        fb=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",13)
        fs=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",11)
    except Exception: fb=fs=ImageFont.load_default()
    if   status!="SUCCESS":  sc,sl=(255,165,0), f"NO_PLAN/{status}"
    elif collapsed:           sc,sl=(180,80,220),"COLLAPSED"
    elif sim_ok:              sc,sl=(80,200,120),f"SUCCESS (sim={sim_lvl}==H{height})"
    else:                     sc,sl=(220,80,60), f"MISMATCH sim={sim_lvl}≠H{height}" if sim_lvl else "SIM ERROR"
    y=_IH+5
    draw.text((8,y),f"[group {group_idx} H{height}] {sl}",fill=sc,font=fb); y+=18
    draw.text((8,y),"✓ known" if matched else "✗ new",
              fill=(160,210,160) if matched else (210,160,160),font=fs); y+=15
    draw.text((8,y),plan_str[:110],fill=(180,180,180),font=fs); y+=15
    draw.text((8,y),group_label[:110],fill=(120,120,120),font=fs)
    if os_swap: y+=15; draw.text((8,y),f"ONE-SHOT: {os_swap[:110]}",fill=(180,130,255),font=fs)
    out = os.path.join(cfg["_img_dir"],
                       re.sub(r"[^\w\-]","_",f"group_{group_idx}_H{height}_{status}"
                              +(f"_os_{re.sub(r'[^\\w]','_',os_swap)}" if os_swap else ""))+".png")
    canvas.save(out)
    return out


# ── NO_PLAN diagnosis ─────────────────────────────────────────────────────────

def write_diagnosis(cfg, group_idx, height, group_label, pair_order,
                    obj_syms, collapse_syms, rel_dim, obj_dim, planner_out):
    dst = os.path.join(cfg["_pddl_dir"],"no_plan_files")
    os.makedirs(dst, exist_ok=True)
    lines = [f"NO_PLAN DIAGNOSIS","="*40,
             f"Group: {group_label}","f idx={group_idx} H={height}",
             f"rel_dim={rel_dim} obj_dim={obj_dim}",
             f"collapse_syms={collapse_syms}","",
             "── Object symbols ──"]
    for k,(key,sym) in enumerate(zip(pair_order,obj_syms)):
        lines.append(f"  obj{k} {key:40s} → {sym}")
    domain_path = cfg["_domain"]
    if os.path.exists(domain_path):
        with open(domain_path) as f: dom = f.read()
        lines += ["","── Domain coverage ──"]
        lines.append(f"  (H{height}) {'found' if f'(H{height})' in dom else 'MISSING'}")
        import re as _re
        acts = _re.findall(r"\(:action.*?(?=\(:action|\Z)",dom,_re.DOTALL)
        lines.append(f"  Total actions: {len(acts)}")
        for k,sym in enumerate(obj_syms):
            preds = [f"(z{j} " if v else f"(not_z{j} " for j,v in enumerate(sym)]
            n_acts = sum(1 for a in acts if all(p in a for p in preds))
            lines.append(f"  obj{k} sym={sym}: {n_acts} actions match"
                         + (" ← NO MATCH" if n_acts==0 else ""))
    prob = os.path.join(cfg["_pddl_dir"],"problem.pddl")
    if os.path.exists(prob):
        with open(prob) as f: lines += ["","── problem.pddl ──", f.read()]
    lines += ["","── Planner output (last 40 lines) ──",
              "\n".join(planner_out.strip().splitlines()[-40:])]
    out = os.path.join(dst, f"group_{group_idx}_H{height}_diagnosis.txt")
    with open(out,"w") as f: f.write("\n".join(lines))
    print(f"  [DIAG] → {out}")


# ── One-shot ──────────────────────────────────────────────────────────────────

def obj_to_canon(otype, path):
    try: return (str(otype), str(float(path)))
    except (ValueError,TypeError):
        return (str(otype), str(path).split("/")[-1].replace(".urdf",""))

def load_match_cache(cfg):
    path = cfg["one_shot"]["match_cache_path"]
    if not os.path.exists(path):
        if cfg["one_shot"]["enabled"]: sys.exit(f"Match cache not found: {path}")
        return {}
    with open(path) as f: return json.load(f)

def build_one_shot_map(cfg, cache):
    if not cache: return {}
    result = defaultdict(list)
    for obj in [tuple(o) for o in cfg["one_shot"].get("objects",[])]:
        canon = obj_to_canon(*obj)
        entry = cache.get(str(canon),{})
        if entry.get("is_novel",True) or entry.get("status")=="already_exists": continue
        try:
            matched = tuple(ast.literal_eval(entry.get("best_match","")))
            result[matched].append(obj)
        except Exception: pass
    return dict(result)

def run_one_shot(cfg, plan_order, pair_to_original, os_map,
                 group_label, n_objs, group_idx, height, plan_str, matched):
    if not os_map: return []
    baseline = [pair_to_original[k] for k in plan_order if pair_to_original.get(k)]
    if len(baseline) != len(plan_order): return []
    canon_to_pos = defaultdict(list)
    for pos,orig in enumerate(baseline):
        c = obj_to_canon(*orig)
        if c in os_map: canon_to_pos[c].append(pos)
    if not canon_to_pos: return []
    z_thr, rows = cfg["eval"]["z_level_threshold"], []
    for orig_canon, positions in canon_to_pos.items():
        for os_obj in os_map[orig_canon]:
            os_canon  = obj_to_canon(*os_obj)
            swap_desc = f"{orig_canon[0]}_{orig_canon[1]} → {os_canon[0]}_{os_canon[1]} pos={positions}"
            print(f"\n  ── [ONE-SHOT] {swap_desc} ──")
            swapped = list(baseline)
            for pos in positions: swapped[pos] = os_obj
            label = f"group_{group_idx}_H{height}_os_{re.sub(r'[^\\w]','_',str(os_canon))}"
            os.makedirs(cfg["_exp_dir"], exist_ok=True)
            import pybullet as _pb
            for _cid in range(20):
                try: _pb.disconnect(_cid)
                except Exception: pass
            try: get_experiment(swapped, cfg["_exp_dir"]+"/", label)
            except Exception as e:
                print(f"  [ONE-SHOT] {e}")
                rows.append({"group":group_label,"n_objects":n_objs,"target_height":height,
                             "status":"ONE_SHOT","sim_success":False,"sim_collapsed":True,
                             "is_one_shot":True,"one_shot_object":str(os_canon),
                             "swapped_original":str(orig_canon),"swap_positions":str(positions),
                             **dict.fromkeys(["plan_order","sim_levels","image_path","planner_output","matched_known_sequence"],None)})
                continue
            csv_p = os.path.join(cfg["_exp_dir"],f"{label}.csv")
            sl, coll = read_sim_levels(csv_p, z_thr)
            ok   = not coll and sl == height
            img  = save_image(cfg, group_label, group_idx, height, plan_str, matched,
                              sl, ok, coll, "SUCCESS", get_rgb_paths(csv_p), os_swap=swap_desc)
            rows.append({"group":group_label,"n_objects":n_objs,"target_height":height,
                         "status":"ONE_SHOT","plan_order":plan_str,"matched_known_sequence":matched,
                         "sim_levels":sl,"sim_success":ok,"sim_collapsed":coll,"image_path":img,
                         "is_one_shot":True,"one_shot_object":str(os_canon),
                         "swapped_original":str(orig_canon),"swap_positions":str(positions),
                         "planner_output":""})
    return rows


# ── Collapse symbols ──────────────────────────────────────────────────────────

def get_collapse_symbols(model, cfg):
    tc  = cfg["_train_cfg"]
    dev = next(model.parameters()).device
    collapse, inserted, normal = get_semantic_symbols(
        model, sym_size=tc["model"]["symbol_size"], out_dim=4,
        collapse_threshold=0.5, device=dev)
    return [list(s) for s in collapse]


# ── Empty row ─────────────────────────────────────────────────────────────────

def _empty(group, n, height, status):
    return {"group":group,"n_objects":n,"target_height":height,"status":status,
            "plan_order":"","matched_known_sequence":False,"sim_levels":None,
            "sim_success":False,"sim_collapsed":None,"image_path":None,
            "is_one_shot":False,"one_shot_object":None,"swapped_original":None,
            "swap_positions":None,"planner_output":""}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config",default="configs/eval.yaml")
    cfg  = load_config(parser.parse_args().config)
    name = cfg["model_name"]
    print(f"\n{'='*60}\n  model : {name}\n  domain: {cfg['_domain']}\n{'='*60}\n")

    if not os.path.exists(cfg["_domain"]):
        sys.exit(f"domain.pddl not found — run learn_rules first for '{name}'")
    os.makedirs(cfg["_pddl_dir"], exist_ok=True)
    os.makedirs(cfg["_eval_dir"], exist_ok=True)

    # 1. Model
    model, train_cfg = load_ckpt(name, tag="best")
    cfg["_train_cfg"] = train_cfg
    _load_data.init(train_cfg)
    set_seed(train_cfg["seed"])
    model.freeze(deterministic=True)
    device  = next(model.parameters()).device
    rel_dim = train_cfg["model"]["symbol_size"]
    obj_dim = train_cfg["model"]["obj_symbol_size"]

    # 2. Collapse symbols
    collapse_syms = get_collapse_symbols(model, cfg)
    print(f"collapse_syms: {collapse_syms}")

    # 3. Config values
    target_min  = cfg["eval"]["target_min_level"]
    target_max  = cfg["eval"]["target_max_level"]
    z_thr       = cfg["eval"]["z_level_threshold"]
    dry_run     = cfg["eval"]["dry_run"]
    one_shot_on = cfg["one_shot"]["enabled"]
    dataset_csv = cfg["data"]["dataset_csv"]

    # 4. Height analysis
    df_det, df_sum = compute_height_analysis(dataset_csv, z_thr)
    df_det["collapsed"] = df_det["collapsed"].astype(bool)
    pair_to_orig = build_pair_to_original(load_csv(dataset_csv))

    # 5. Filter groups
    targets = df_sum[(df_sum["min_levels"]==target_min) &
                     (df_sum["max_levels"]==target_max)].reset_index(drop=True)
    print(f"Groups with min={target_min} & max={target_max}: {len(targets)}")
    if targets.empty: return

    # 6. Object symbols
    all_unique_pairs = set()
    for _, grow in targets.iterrows():
        try: all_unique_pairs.update(ast.literal_eval(grow["object_group"]))
        except Exception: pass
    all_pairs    = sorted(all_unique_pairs)
    all_inputs   = [pair_to_orig[k] for k in all_pairs if k in pair_to_orig]
    all_syms     = get_obj_symbols(all_inputs, model, cfg)
    sym_cache = {}
    for key, sym in zip(all_pairs, all_syms):
        sym_cache[key] = sym
        print(f"  {key} → {sym}")

    # 7. One-shot setup
    os_map = build_one_shot_map(cfg, load_match_cache(cfg)) if one_shot_on else {}

    results = []

    # 8. Main loop
    for i, grow in targets.iterrows():
        group_label = grow["object_group"]
        n_objs      = int(grow["n_objects"])
        print(f"\n{'='*60}\n[{i+1}/{len(targets)}] {group_label}")

        objs, pair_order = group_to_objects(group_label, pair_to_orig)
        if objs is None:
            for h in [target_min, target_max]:
                results.append(_empty(group_label, n_objs, h, "SKIP_PARSE_ERROR"))
            continue

        obj_syms = [sym_cache[k] for k in pair_order if k in sym_cache]
        if len(obj_syms) != len(pair_order):
            for h in [target_min, target_max]:
                results.append(_empty(group_label, n_objs, h, "SKIP_SYMBOL_ERROR"))
            continue

        for height in [target_min, target_max]:
            print(f"\n  ── H{height} ──")
            known = known_sequences(df_det, group_label, height)

            try:
                write_problem_pddl(obj_syms, collapse_syms, [], rel_dim, obj_dim,
                                   height, cfg["_pddl_dir"])
            except Exception as e:
                results.append(_empty(group_label, n_objs, height, f"PDDL_ERROR:{e}"))
                continue

            if dry_run:
                shutil.copy(os.path.join(cfg["_pddl_dir"],"problem.pddl"),
                            os.path.join(cfg["_pddl_dir"],f"problem_H{height}_dry.pddl"))
                continue

            status, plan_lines, full_out = run_planner(cfg)
            save_pddl_copy(cfg, i, height, status)
            print(f"  Planner: {status}")

            plan_str, matched = "", False
            sim_lvl = sim_ok = sim_coll = img_path = None
            sim_ok = False; sim_coll = None

            if status == "SUCCESS" and plan_lines:
                order = extract_order(plan_lines, pair_order)
                if order:
                    plan_str = " > ".join(order)
                    matched  = order in known
                    print(f"  Plan: {plan_str}  matched={matched}")
                    sim_lvl, sim_ok, sim_coll, sim_csv = run_simulation(
                        order, pair_to_orig, cfg, i, height)
                    rgb = get_rgb_paths(sim_csv) if sim_csv else []
                    img_path = save_image(cfg, group_label, i, height,
                                         plan_str, matched, sim_lvl, sim_ok,
                                         sim_coll, status, rgb)
                    if one_shot_on:
                        results.extend(run_one_shot(cfg, order, pair_to_orig, os_map,
                                                    group_label, n_objs, i, height,
                                                    plan_str, matched))
            else:
                img_path = save_image(cfg, group_label, i, height,
                                      "", False, None, False, None, status, [])
                if status in ("NO_PLAN",) or status.startswith("ERROR_"):
                    write_diagnosis(cfg, i, height, group_label, pair_order,
                                    obj_syms, collapse_syms, rel_dim, obj_dim, full_out)

            results.append({"group":group_label,"n_objects":n_objs,
                            "target_height":height,"status":status,
                            "plan_order":plan_str,"matched_known_sequence":matched,
                            "sim_levels":sim_lvl,"sim_success":sim_ok,
                            "sim_collapsed":sim_coll,"image_path":img_path,
                            "is_one_shot":False,"one_shot_object":None,
                            "swapped_original":None,"swap_positions":None,
                            "planner_output":full_out[:600]})
        if dry_run: break

    # 9. Save + summary
    if dry_run or not results:
        print("[DRY_RUN] done"); return
    df_res = pd.DataFrame(results)
    df_res.to_csv(cfg["_results"], index=False)
    df_base = df_res[~df_res["is_one_shot"]]
    print(f"\n{'='*60}\nFINAL SUMMARY")
    for h in [target_min, target_max]:
        sub  = df_base[df_base["target_height"]==h]
        n_ok = (sub["status"]=="SUCCESS").sum()
        n_r  = sub["sim_levels"].notna().sum()
        n_s  = sub["sim_success"].sum()
        pct  = f"{100*n_s//n_r}%" if n_r else "—"
        print(f"  H{h}: plan={n_ok}/{len(sub)}  sim={n_s}/{n_r} ({pct})")
    df_os = df_res[df_res["is_one_shot"]==True]
    if not df_os.empty:
        print(f"\n── ONE-SHOT ──")
        for osl in sorted(df_os["one_shot_object"].dropna().unique()):
            for h in [target_min, target_max]:
                sub = df_os[(df_os["one_shot_object"]==osl)&(df_os["target_height"]==h)]
                if sub.empty: continue
                n_r = sub["sim_levels"].notna().sum()
                n_s = sub["sim_success"].sum()
                print(f"  [{osl}] H{h}: {n_s}/{n_r}")
    print(f"\nSaved → {cfg['_results']}")


if __name__ == "__main__":
    main()