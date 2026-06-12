import argparse
import os
import pickle
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from collections import namedtuple, defaultdict, Counter

from tqdm import tqdm
from model import load_ckpt, save_clusters
from classes.state import State
from classes.effect import Effect
from load_data import preprocess_to_dataset_normalized, set_seed, calculate_bbox_stats, load_data_pddl
import load_data as _load_data
from utils import build_bank_optimized
from symbol_semantics_fast import get_semantic_symbols, create_collapse_checks
# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
# One sample = one tower placement step, analogous to RelDeepSym's
# (state, action, next_state) tuple.
#
# Differences from RelDeepSym:
#   1. state includes ACCUMULATED relations from all previous steps
#      (not a single-shot snapshot), because each placement builds on history.
#   2. action is the object symbol o_i of the newly placed object
#      (not a continuous action vector).
#   3. No mask — towers have variable size, no padding needed.
#   4. Samples within a tower are NOT independent: r_i at step k
#      comes from r_f of step k-1 of the same tower.
#
#   state      = (z_i, r_i)   objects + accumulated relations BEFORE placement
#   action     = o_i           symbol of newly placed object
#   next_state = (z_f, r_f)   objects + accumulated relations AFTER placement

Sample = namedtuple("Sample", [
    "z_i",    # [n_existing,   obj_sym]  object symbols before placement
    "r_i",    # [n_pairs_i,    rel_sym]  accumulated relations before placement
    "o_i",    # [1,            obj_sym]  symbol of newly placed object (= action)
    "z_f",    # [n_existing+1, obj_sym]  object symbols after placement
    "r_f",    # [n_pairs_f,    rel_sym]  accumulated relations after placement
    "z_all",  # [n_existing+1, obj_sym]  same as z_f, kept for compatibility
])


class TowerDataset(torch.utils.data.Dataset):
    """Flat list of Sample namedtuples, one per (tower, step)."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



# N_EPS helpers
def _hash_tensor(t):
    return str(t.cpu().tolist())


def _vote(tensor_list):
    """
    Majority vote over N_EPS versions of the same tensor.
    Analogous to RelDeepSym's mean+round, but over full symbol vectors
    (no per-bit voting — each relation/object symbol is an atomic unit).
    Returns (winner_tensor, Counter of pattern -> count).
    """
    hashes      = [_hash_tensor(t) for t in tensor_list]
    counts      = Counter(hashes)
    winner_hash = counts.most_common(1)[0][0]
    winner      = next(t for t, h in zip(tensor_list, hashes) if h == winner_hash)
    return winner, counts


def _repeat_graph(single, n_eps, device):
    """
    Repeat a single PyG Data object n_eps times into one batched graph,
    exactly like RelDeepSym's state.unsqueeze(0).repeat(N_EPS, ...).
    """
    return Batch.from_data_list([single] * n_eps)


# collate_preds

def _encode_step_batched(model, single, single_given, single_new_obj,
                         n_eps, device, chunk_size=100):
    """
    Same as before but processes n_eps in chunks to avoid OOM.
    """
    n_nodes = single.num_nodes
    n_given = n_nodes - 1

    all_z     = []
    all_zn    = []
    all_r_new = []

    remaining = n_eps
    while remaining > 0:
        chunk = min(chunk_size, remaining)
        remaining -= chunk

        repeated   = _repeat_graph(single, chunk, device)
        repeated.x = repeated.x.float()

        offsets     = torch.arange(chunk, device=device) * n_nodes
        given_rep   = (single_given.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        new_obj_rep = (single_new_obj.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)

        bits_rep, _ = model.get_object_symbols(repeated.x)
        _, r_new_rep = model(repeated.x, repeated.edge_index,
                             repeated.edge_attr,
                             given_rep, new_obj_rep)

        rel_sym = r_new_rep.shape[-1]

        bits_rep  = bits_rep.reshape(chunk, n_nodes, -1).cpu()
        r_new_rep = r_new_rep.reshape(chunk, n_given, rel_sym).cpu()

        all_z.append(bits_rep[:, :n_given, :])
        all_zn.append(bits_rep)
        all_r_new.append(r_new_rep)

    z_all     = torch.cat(all_z,     dim=0)   # [n_eps, n_given, obj_sym]
    zn_all    = torch.cat(all_zn,    dim=0)   # [n_eps, n_nodes, obj_sym]
    r_new_all = torch.cat(all_r_new, dim=0)   # [n_eps, n_given, rel_sym]

    return z_all, zn_all, r_new_all

def collate_preds(model, loader, n_eps=100):
    """
    Build a TowerDataset by running N_EPS stochastic passes per tower step.

    Mirrors RelDeepSym's collate_preds structure:
      - RelDeepSym: repeat each (state, next_state) N_EPS times in batch dim,
                    encode all at once, vote via mean+round.
      - Here:       repeat each tower step N_EPS times in batch dim,
                    single forward call per step, vote per component
                    (z, zn, r_new) independently.
                    Voted winner carried forward as r_i/z_i for next step
                    to maintain chain consistency.

    Returns a TowerDataset (flat list of Sample namedtuples).
    """
    model.eval()                          # ← FIXED: BatchNorm uses running stats
    model.gs_layer.deterministic = False  # keep stochastic for Gumbel voting
    torch.set_grad_enabled(False)
    device = next(model.parameters()).device

    print("→ Grouping batches by tower...")
    tower_batches = defaultdict(list)
    for batch in loader:
        for i in range(batch.num_graphs):
            single_cpu = batch.get_example(i)
            t_id  = single_cpu.id
            step  = int(single_cpu.step)
            tower_batches[t_id].append((step, single_cpu))

    for t_id in tower_batches:
        tower_batches[t_id].sort(key=lambda x: x[0])
    n_towers = len(tower_batches)
    print(f"→ {n_towers} towers found. Running {n_eps} passes per step (batched)...")

    # ── per-tower, per-step voting ────────────────────────────────────────
    all_samples = []
    all_reports = {}

    for t_id, steps in tqdm(tower_batches.items(), total=n_towers):

        # chain state carried from voted winner of previous step
        prev_r_f = torch.empty(0)
        prev_z_f = None

        tower_reports = []

        for (step_val, single_cpu) in steps:
            single = single_cpu.to(device)
            step_val = int(single.step)
            n_nodes  = single.num_nodes
            n_given  = n_nodes - 1

            single_given   = torch.arange(n_given, device=device)
            single_new_obj = torch.full((n_given,), n_given, device=device)
            single.x       = single.x.float()

            # ── N_EPS batched stochastic passes ───────────────────────────
            # One forward call processes all n_eps copies simultaneously.
            # Analogous to RelDeepSym's single batched encode over N_EPS repeats.
            with torch.no_grad():
                z_all, zn_all, r_new_all = _encode_step_batched(
                    model, single, single_given, single_new_obj, n_eps, device, chunk_size=200   # tune this
                )
            # z_all    : [n_eps, n_given, obj_sym]
            # zn_all   : [n_eps, n_nodes, obj_sym]
            # r_new_all: [n_eps, n_given, rel_sym]
            torch.cuda.empty_cache()
            # ── vote independently on each component ──────────────────────
            winner_z,     z_counts     = _vote([z_all[k]     for k in range(n_eps)])
            winner_zn,    zn_counts    = _vote([zn_all[k]    for k in range(n_eps)])
            winner_r_new, r_new_counts = _vote([r_new_all[k] for k in range(n_eps)])

            # ── assemble Sample ───────────────────────────────────────────
            # z_i and r_i are FIXED from previous voted winner (chain anchor)
            z_i = prev_z_f if prev_z_f is not None else winner_z
            r_i = prev_r_f
            r_f = torch.cat([r_i, winner_r_new], dim=0) if r_i.numel() > 0 \
                  else winner_r_new
            z_f = winner_zn
            o_i = winner_zn[-1].unsqueeze(0)   # last node = newly placed object

            all_samples.append(Sample(
                z_i   = z_i,
                r_i   = r_i,
                o_i   = o_i,
                z_f   = z_f,
                r_f   = r_f,
                z_all = z_f,
            ))

            # ── carry voted winner forward as anchor for next step ────────
            prev_z_f = z_f
            prev_r_f = r_f

            # ── store for report ──────────────────────────────────────────
            tower_reports.append((
                step_val,
                z_counts, zn_counts, r_new_counts,
            ))

        all_reports[t_id] = tower_reports

    _print_report(all_reports, n_towers, n_eps)

    return TowerDataset(all_samples)


def _print_report(all_reports, n_towers, n_eps):
    total = sum(len(reports) for reports in all_reports.values())
    print(f"→ Voting complete: {total} steps across {n_towers} towers, {n_eps} passes each.")


def compute_operators(trainloader, min_dominance=0.0):
    preconditions = {}
    print("→ Identifying unique preconditions...")
    for t, sample in enumerate(tqdm(trainloader)):
        z_i, r_i, o_i, z_f, r_f, _ = sample
        z_i = z_i.squeeze(0)
        r_i = r_i.squeeze(0)
        o_i = o_i.squeeze(0)
        z_f = z_f.squeeze(0)
        r_f = r_f.squeeze(0)
        z_gr = State(z_f, z_i, r_i, cfg, act_symbol=o_i)
        names = {i: f"o{i}" for i in range(len(z_gr.get_params()))}
        z_abs = z_gr.substitute(names)
        if z_abs in preconditions:
            preconditions[z_abs].append((t, names))
        else:
            preconditions[z_abs] = [(t, names)]
        if (t + 1) % 100 == 0:
            print(f"   {t+1} samples — {len(preconditions)} preconditions")
    print(f"→ {len(preconditions)} unique preconditions found.")

    dataset = trainloader.dataset
    print("→ Identifying unique effects per precondition...")

    all_operators = {}
    sorted_pre = sorted(preconditions.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (precond, trans_list) in enumerate(tqdm(sorted_pre)):
        effects = {}
        for (idx, names) in trans_list:
            s = dataset[idx]
            eff_abs = Effect(s.z_i, s.r_i, s.z_f, s.r_f, cfg).substitute(names)
            effects[eff_abs] = effects.get(eff_abs, 0) + 1
        all_operators[precond] = effects

    SEP     = "─" * 65
    n_total  = len(all_operators)
    n_single = sum(1 for e in all_operators.values() if len(e) == 1)
    n_multi  = sum(1 for e in all_operators.values() if len(e) > 1)
    print(f"── Effect Diversity Report ({n_total} preconditions) ──")
    print(f"  Single effect   : {n_single:4d} ({100*n_single/max(n_total,1):.1f}%)")
    print(f"  Multiple effects: {n_multi:4d} ({100*n_multi/max(n_total,1):.1f}%)")
    for op_precond, effects in sorted(all_operators.items(),
                                      key=lambda x: len(x[1]), reverse=True):
        if len(effects) == 1:
            break
        total        = sum(effects.values())
        counts       = sorted(effects.values(), reverse=True)
        pcts         = [f"{100*c/total:.1f}%" for c in counts]
        dominant_pct = 100 * counts[0] / total
        print(f"  {len(effects)} effects | total={total:5d} | "
              f"dominant={dominant_pct:.1f}% | dist={pcts}")
    print(SEP)

    operators = {}
    skipped   = 0
    for precond, effects in all_operators.items():
        total          = sum(effects.values())
        dominant_eff   = max(effects, key=effects.get)
        dominant_cnt   = effects[dominant_eff]
        dominant_ratio = dominant_cnt / total
        if dominant_ratio < min_dominance:
            skipped += 1
            continue
        operators[precond] = {dominant_eff: dominant_cnt}

    print(f"→ {len(operators)} operators selected (dominant effect strategy).")
    if skipped:
        print(f"   {skipped} preconditions skipped (dominance < {min_dominance:.0%}).")
    return operators


def precond_to_pddl(params, z_i, r_i, indentation="\t\t"):
    schema = f"(and\n{indentation}"
    for i, p1 in enumerate(params):
        for p2 in params[i+1:]:
            schema += f"(not (= {p1} {p2})) "
    schema += "\n"

    obj_size     = len(next(iter(z_i.values()))) - 1
    active_count = sum(1 for v in z_i.values() if len(v) > obj_size and v[obj_size] == 1)

    schema += f"{indentation}(active-count-{active_count})\n"
    schema += f"{indentation}(not (pending-collapse-check))\n"
    schema += f"{indentation}(not (pending-height-check))\n"
    schema += f"{indentation}"
    for i, p in enumerate(reversed(params[:active_count])):
        schema += f"(top-{i+1} {p}) "
    schema += "\n"

    for name, obj_val in z_i.items():
        if obj_val:
            schema += indentation
            for j, val in enumerate(obj_val):
                schema += f"(z{j} ?{name}) " if val == 1 else f"(not (z{j} ?{name})) "
            schema += "\n"

    for k, rel_dict in enumerate(r_i):
        if rel_dict:
            schema += indentation
            for (n1, n2), val in rel_dict.items():
                schema += f"(r{k} ?{n1} ?{n2}) " if val == 1 else f"(not (r{k} ?{n1} ?{n2})) "
            schema += "\n"

    schema += "\t)\n"
    return schema, active_count


def effect_to_pddl(effect, active_count, params, indentation="\t\t"):
    # Always deterministic — pick the dominant (most frequent) effect.
    dominant = max(effect, key=effect.get)
    z_eff, r_eff = dominant.z_eff, dominant.r_eff

    schema = "(and\n"
    if active_count == 3:
        schema += f"{indentation}(has-top-4)\n"
    else:
        schema += f"{indentation}(not (active-count-{active_count})) (active-count-{active_count+1})\n"
    schema += f"{indentation}(pending-collapse-check)\n"
    schema += f"{indentation}(pending-height-check)\n"

    top_items = list(reversed(params)) if active_count < 3 else list(reversed(params[-active_count-1:]))
    for i, p in enumerate(top_items):
        schema += f"{indentation}(and (top-{i+1} {p}) (not (top-{i} {p})))\n"

    _ZVAL = {1:  "(z{j} ?{n}) ",
            -1: "(not (z{j} ?{n})) "}
    for name, obj_val in z_eff.items():
        if obj_val:
            schema += indentation
            for j, val in enumerate(obj_val):
                if val in _ZVAL:
                    schema += _ZVAL[val].format(j=j, n=name)
            schema += "\n"

    _RVAL = {1:  "(r{k} ?{n1} ?{n2}) ",
            -1: "(not (r{k} ?{n1} ?{n2})) "}
    for k, rel_dict in enumerate(r_eff):
        if rel_dict:
            schema += indentation
            for (n1, n2), val in rel_dict.items():
                if val in _RVAL:
                    schema += _RVAL[val].format(k=k, n1=n1, n2=n2)
            schema += "\n"

    schema += f"{indentation[1:]})\n"
    return schema


def operator_to_pddl(idx, precond, effect):
    params = [f"?{p}" for p in precond.get_params()]
    precond_pddl, active_count = precond_to_pddl(params, precond.obj_dict, precond.relations)
    effect_pddl  = effect_to_pddl(effect, active_count, params)
    count        = max(effect.values())
    action_name  = "_".join(map(str, precond.action))
    schema  = f"(:action a_{action_name}_i{idx}_c{count}\n"
    schema += f"\t:parameters ({' '.join(params)})\n"
    schema += f"\t:precondition {precond_pddl}"
    schema += f"\t:effect {effect_pddl}"
    schema += ")\n"
    return schema


def create_init_actions(operators, latent_dim):
    """One init action per unique first-object symbol seen across all z_f."""
    seen   = set()
    unique = []
    for precond in operators:
        for name, bits in precond.obj_dict.items():
            key = tuple(bits[:latent_dim])
            if key not in seen:
                seen.add(key)
                unique.append(bits[:latent_dim])

    init_actions = []
    for idx, obj_bits in enumerate(unique):
        bits_str    = "_".join(map(str, obj_bits))
        schema  = f"(:action a_{bits_str}_i{idx}\n"
        schema += f"\t:parameters (?o0)\n"
        schema += f"\t:precondition (and\n"
        schema += f"\t\t(active-count-0)\n"
        schema += f"\t\t(not (pending-collapse-check))\n"
        schema += f"\t\t(not (pending-height-check))\n"
        schema += f"\t\t(top-0 ?o0)\n\t\t"
        for j, val in enumerate(obj_bits):
            schema += f"(z{j} ?o0) " if val == 1 else f"(not (z{j} ?o0)) "
        schema += f"(not (z{latent_dim} ?o0))\n"
        schema += f"\t)\n"
        schema += f"\t:effect (and\n"
        schema += f"\t\t(not (active-count-0)) (active-count-1)\n"
        schema += f"\t\t(pending-height-check)\n"
        schema += f"\t\t(z{latent_dim} ?o0)\n"
        schema += f"\t\t(top-1 ?o0)\n"
        schema += f"\t)\n)\n"
        init_actions.append(schema)
    return init_actions


def create_final(latent_dim):
    schema  = f"(:action conclude\n"
    schema += f"\t:parameters ()\n"
    schema += f"\t:precondition (and\n"
    schema += f"\t\t(active-count-3)\n"
    schema += f"\t\t(not (pending-collapse-check))\n"
    schema += f"\t\t(not (pending-height-check))\n"
    schema += f"\t\t(forall (?o) (z{latent_dim} ?o))\n"
    schema += f"\t)\n"
    schema += f"\t:effect (and\n"
    schema += f"\t\t(not (active-count-3)) (all-used)\n"
    schema += f"\t)\n)\n"
    return schema

def construct_domain(init_actions, action_schemas, final_schema, collapse_schema,
                     latent_dim, relation_dim, max_count=10, last_x=4):
    domain  = "(define (domain blocks)\n"
    domain += "\t(:requirements :equality :disjunctive-preconditions :universal-preconditions)\n"
    domain += "\t(:predicates\n"
    for i in range(latent_dim + 1):      # +1 for the active bit
        domain += f"\t\t(z{i} ?x)\n"
    for i in range(relation_dim + 1):
        domain += f"\t\t(r{i} ?x ?y)\n"
    for i in range(max_count + 1):
        domain += f"\t\t(active-count-{i})\n"
        domain += f"\t\t(H{i})\n"
    for i in range(last_x + 1):
        domain += f"\t\t(top-{i} ?x)\n"
    for i in range(1, max_count + 1):
        domain += f"\t\t(check_collapse_count_{i})\n"
    domain += "\t\t(all-used)\n\t\t(pending-collapse-check)\n\t\t(pending-height-check)\n"
    domain += "\t\t(active-count-collapse)\n\t\t(newly-added ?x)\n\t\t(has-top-4)\n\t\t(stack)\n"
    domain += "\t)\n"
    for schema in action_schemas:
        domain += schema
    for schema in init_actions:
        domain += schema
    domain += final_schema
    domain += collapse_schema
    domain += ")"
    return domain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",             type=str,   required=True)
    parser.add_argument("-min_dominance", type=float, default=0.0)
    parser.add_argument("-n_eps",         type=int,   default=500)
    args = parser.parse_args()

    model, cfg = load_ckpt(args.n, tag="best")
    model.freeze(deterministic=False)
    _load_data.init(cfg)
    set_seed(cfg["seed"])
    device = next(model.parameters()).device

    # ── Auto-fit clusters if centroids were not saved yet ─────────────────
    if model.cluster_centroids.numel() == 0:
        print("\n⚠️  No saved centroids — fitting clusters now over all splits...")
        csv_path  = cfg["data"]["csv_path"]
        depth_dir = cfg["data"]["depth_image_dir"]
        train_rows, val_rows = load_data_pddl(csv_path)
        all_rows = train_rows + val_rows
        STATS_FOR_NORM = calculate_bbox_stats(train_rows)
        imgs = build_bank_optimized(all_rows, depth_dir)
        dataset = preprocess_to_dataset_normalized(all_rows, imgs, torch.device("cpu"), STATS_FOR_NORM)
        fit_loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                                shuffle=False, num_workers=0)
        model.fit_clusters(fit_loader)
        save_clusters(model, args.n, tag="best")
        print("✅ Centroids fitted and saved.\n")

    save_path = os.path.join("save", args.n)
    os.makedirs(save_path, exist_ok=True)

    dataset_path = os.path.join(save_path, "trainset.pt")
    if not os.path.exists(dataset_path):
        csv_path  = cfg["data"]["csv_path"]
        depth_dir = cfg["data"]["depth_image_dir"]
        train_rows, val_rows = load_data_pddl(csv_path)
        train_rows = train_rows + val_rows
        STATS_FOR_NORM = calculate_bbox_stats(train_rows)
        imgs    = build_bank_optimized(train_rows, depth_dir)
        dataset = preprocess_to_dataset_normalized(train_rows, imgs, torch.device("cpu"), STATS_FOR_NORM)
        loader  = DataLoader(dataset, batch_size=cfg["training"]["batch_size"],
                             shuffle=False, num_workers=0)

        tower_dataset = collate_preds(model, loader, n_eps=args.n_eps)
        torch.save(tower_dataset, dataset_path)
        print(f"→ Saved {len(tower_dataset)} samples to {dataset_path}")
    else:
        tower_dataset = torch.load(dataset_path, weights_only=False)
        print(f"→ Loaded {len(tower_dataset)} samples from {dataset_path}")

    trainloader = torch.utils.data.DataLoader(tower_dataset, batch_size=1)

    operator_path = os.path.join(save_path, "operators.pkl")
    if not os.path.exists(operator_path):
        operators = compute_operators(trainloader, args.min_dominance)
        pickle.dump(operators, open(operator_path, "wb"))
    else:
        operators = pickle.load(open(operator_path, "rb"))
        print(f"→ Loaded operators from {operator_path}")


    obj_dim = cfg["model"]["obj_symbol_size"]
    rel_dim = cfg["model"]["symbol_size"]

    action_schemas = [
        operator_to_pddl(i, precond, effects)
        for i, (precond, effects) in enumerate(operators.items())
        if sum(effects.values()) > 0
    ]

    collapse_syms, inserted_syms, normal_sym = get_semantic_symbols(model, rel_dim, device)
    collapse_schema = create_collapse_checks(collapse_syms, inserted_syms, normal_sym, sym_size=obj_dim)

    init_actions  = create_init_actions(operators, latent_dim=obj_dim)
    final_schema  = create_final(latent_dim=obj_dim, max_objs=4)
    domain        = construct_domain(init_actions, action_schemas, final_schema, collapse_schema,
                                     latent_dim=obj_dim, relation_dim=rel_dim)

    domain_path = os.path.join(save_path, "domain.pddl")
    with open(domain_path, "w") as f:
        f.write(domain)
    print(f"→ Constructed domain with {len(action_schemas)} action schemas → {domain_path}")