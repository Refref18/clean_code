"""
symbol_semantics.py
===================
Queries the trained model's decoder to determine which relation symbol patterns
correspond to:
  - COLLAPSE  : tower falls (collapse_flag >= threshold)
  - INSERTED  : object is on-top / inside — does NOT add stack height
  - NORMAL    : plain stacking — adds stack height

These are then used by create_collapse_checks() and create_height_actions() so
that both functions work correctly regardless of which model is loaded, instead
of assuming fixed indices (e.g. r0=collapse, r1=on_top).

Usage
-----
    from symbol_semantics import get_semantic_symbols, create_collapse_checks, create_height_actions

    collapse_syms, inserted_syms, normal_sym = get_semantic_symbols(model)
    collapse_schema = create_collapse_checks(collapse_syms, inserted_syms, normal_sym, sym_size=SYM_SIZE)
"""
# TODO: right now it only considers the largest zmax as on top 
import itertools
import torch
# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — build the full symbol → decoded-effect map from the model
# ─────────────────────────────────────────────────────────────────────────────

def get_symbol_map(model, sym_size=None, out_dim=None, device="cpu"):
    """
    Enumerate every possible binary symbol vector (2^sym_size patterns),
    pass each through model.decoder, apply sigmoid to the collapse dimension,
    and return a dict:

        { (bit0, bit1, ...): decoded_effect_tensor }

    This is identical to the version in compound_dictionary_w_step1_vis_coll.py
    but accepts sym_size / out_dim overrides so it works with any model config.
    """

    symbol_map = {}
    model.eval()
    with torch.no_grad():
        for combo in itertools.product([0.0, 1.0], repeat=sym_size):
            sym = torch.tensor(combo, device=device, dtype=torch.float32)
            out = model.decoder(sym).clone()
            out[out_dim] = torch.sigmoid(out[out_dim])   # collapse probability
            symbol_map[combo] = out
    return symbol_map


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — classify each symbol as collapse / inserted / normal
# ─────────────────────────────────────────────────────────────────────────────

def get_semantic_symbols(model, sym_size=None, device="cpu", out_dim=4, collapse_threshold=0.5):
    """
    Determines the semantic role of every relation symbol by decoding it through
    the trained model and inspecting the resulting effect values.

    Semantics
    ---------
    COLLAPSE  — decoded collapse_flag >= collapse_threshold
                → the tower fell; triggers active-count-collapse in PDDL

    NORMAL    — non-collapse symbol with the HIGHEST MaxZ (index 3 of the effect)
                → plain stacking; the new object sits cleanly on top and
                  the height counter should increase

    INSERTED  — all remaining non-collapse symbols
                → object ended up inside something, or on-top in a way that
                  doesn't contribute a full height increment (e.g. cup-on-cup)

    Parameters
    ----------
    model              : trained TGModel (must have a .decoder attribute)
    sym_size           : number of relation bits (default: SYM_SIZE from variables)
    out_dim            : index of the collapse flag in the decoded effect (default: OUT_DIM)
    collapse_threshold : sigmoid score above which a symbol is considered collapse

    Returns
    -------
    collapse_syms  : list of list[int]  —  e.g. [[1,0,0,1,1,0], ...]
    inserted_syms  : list of list[int]  —  e.g. [[0,1,0,0,0,1], ...]
    normal_sym     : list[int]          —  e.g. [0,0,1,0,1,1]
    """
    symbol_map    = get_symbol_map(model, sym_size=sym_size, out_dim=out_dim, device=device)
    collapse_syms = []
    non_collapse  = []   # list of (bits_list, max_z_value)

    for combo, effect in symbol_map.items():
        bits          = [int(x) for x in combo]
        collapse_prob = effect[out_dim].item()

        if collapse_prob >= collapse_threshold:
            collapse_syms.append(bits)
        else:
            # Index 3 in the decoded effect corresponds to MaxZ
            # (order: MinX=0, MinZ=1, MaxX=2, MaxZ=3 — matching STATS_FOR_NORM)
            max_z = effect[3].item()
            non_collapse.append((bits, max_z))

    if not non_collapse:
        raise RuntimeError(
            "All symbols were classified as collapse — check collapse_threshold "
            f"(currently {collapse_threshold}) or verify the model decoder."
        )

    # Normal stacking = the non-collapse symbol whose decoded MaxZ is highest
    # (placing an object on top causes the largest positive upward displacement)
    non_collapse_sorted = sorted(non_collapse, key=lambda x: x[1], reverse=True)
    normal_sym          = non_collapse_sorted[0][0]
    inserted_syms       = [bits for bits, _ in non_collapse_sorted[1:]]

    print("=" * 60)
    print("  SEMANTIC SYMBOL CLASSIFICATION")
    print("=" * 60)
    print(f"  sym_size         : {sym_size}")
    print(f"  out_dim (collapse bit) : {out_dim}")
    print(f"  collapse_threshold     : {collapse_threshold}")
    print(f"  collapse  symbols ({len(collapse_syms)}) : {collapse_syms}")
    print(f"  normal    symbol       : {normal_sym}  (MaxZ = {non_collapse_sorted[0][1]:.4f})")
    print(f"  inserted  symbols ({len(inserted_syms)}) : {inserted_syms}")
    print("=" * 60)

    return collapse_syms, inserted_syms, normal_sym


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PDDL helper: convert a bit pattern to a predicate string
# ─────────────────────────────────────────────────────────────────────────────

def bits_to_pddl_pair(bits, var1, var2, indent="\t\t\t\t"):
    """
    Given a binary list like [1, 0, 1, 0, 0, 1] and two PDDL variable names,
    returns a PDDL conjunction string, e.g.:
        (and (r0 ?o0 ?o1) (not_r1 ?o0 ?o1) (r2 ?o0 ?o1) ...)
    """
    terms = []
    for i, bit in enumerate(bits):
        if bit == 1:
            terms.append(f"(r{i} {var1} {var2})")
        else:
            terms.append(f"(not (r{i} {var1} {var2}))")
    return "(and " + " ".join(terms) + ")"


def syms_to_pddl_or(sym_list, var1, var2, indent="\t\t\t"):
    """
    Wraps a list of bit patterns in an (or ...) block.
    If only one pattern, returns a plain (and ...) without the outer (or ...).
    """
    conjunctions = [bits_to_pddl_pair(bits, var1, var2) for bits in sym_list]
    if len(conjunctions) == 1:
        return f"{indent}{conjunctions[0]}\n"
    out = f"{indent}(or\n"
    for c in conjunctions:
        out += f"{indent}\t{c}\n"
    out += f"{indent})\n"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — re-implemented PDDL generators that accept semantic symbols
# ─────────────────────────────────────────────────────────────────────────────

def create_collapse_checks(collapse_syms, inserted_syms, normal_sym, sym_size=None):
    """
    Generates the collapse-check and no-collapse PDDL actions.

    Replaces the hardcoded version in DOMAIN_new_model.py by using the actual
    collapse / non-collapse symbol patterns from the model.

    Parameters
    ----------
    collapse_syms  : list of list[int]  (from get_semantic_symbols)
    inserted_syms  : list of list[int]  (from get_semantic_symbols)
    normal_sym     : list[int]          (from get_semantic_symbols)
    sym_size       : int, number of relation bits (used for documentation only)

    Returns
    -------
    str  — PDDL action definitions
    """
    actions = ""

    # ── check_collapse_count_{count}  (tower fell) ───────────────────────────
    for count in range(2, 4):
        newest = count - 1
        params = " ".join([f"?o{i}" for i in range(count)])

        actions += f"(:action check_collapse_count_{count}\n"
        actions += f"\t:parameters ({params})\n"
        actions += f"\t:precondition (and\n"
        actions += f"\t\t(active-count-{count})\n"
        actions += f"\t\t(pending-collapse-check)\n"
        # Any pair involving the newest object must match a collapse symbol
        actions += f"\t\t(or\n"
        for i in range(newest):
            v1, v2 = f"?o{i}", f"?o{newest}"
            for csym in collapse_syms:
                actions += f"\t\t\t{bits_to_pddl_pair(csym, v1, v2)}\n"
        actions += f"\t\t)\n"
        actions += f"\t)\n"
        actions += f"\t:effect (and\n"
        actions += f"\t\t(not (active-count-{count}))\n"
        actions += f"\t\t(not (pending-collapse-check))\n"
        actions += f"\t\t(active-count-collapse)\n"
        actions += f"\t)\n"
        actions += ")\n\n"

    # ── no_collapse_count_{count}  (tower is safe to continue) ───────────────
    # Safe = no pair with the newest object matches any collapse symbol.
    # We express this as: for every pair, at least one bit differs from every
    # collapse pattern (i.e. the pair matches a non-collapse symbol instead).
    non_collapse_syms = inserted_syms + [normal_sym]

    for count in range(2, 4):
        newest = count - 1
        params = " ".join([f"?o{i}" for i in range(count)])

        actions += f"(:action no_collapse_count_{count}\n"
        actions += f"\t:parameters ({params})\n"
        actions += f"\t:precondition (and\n"
        actions += f"\t\t(active-count-{count})\n"
        actions += f"\t\t(pending-collapse-check)\n"

        # Pin objects to stack positions
        for i in range(count):
            stack_pos = count - i
            actions += f"\t\t(top-{stack_pos} ?o{i})\n"

        # Distinctness
        for i in range(count):
            for j in range(i + 1, count):
                actions += f"\t\t(not (= ?o{i} ?o{j}))\n"

        # Each pair with the newest must match one of the non-collapse symbols
        for i in range(newest):
            v1, v2 = f"?o{i}", f"?o{newest}"
            actions += f"\t\t(or\n"
            for ncsym in non_collapse_syms:
                actions += f"\t\t\t{bits_to_pddl_pair(ncsym, v1, v2)}\n"
            actions += f"\t\t)\n"

        actions += f"\t)\n"
        actions += f"\t:effect (and\n"
        actions += f"\t\t(not (pending-collapse-check))\n"
        actions += f"\t)\n"
        actions += ")\n\n"

    # ── height actions follow ─────────────────────────────────────────────────
    actions += create_height_actions(collapse_syms, inserted_syms, normal_sym, sym_size=sym_size)
    return actions


def create_height_actions(collapse_syms, inserted_syms, normal_sym, sym_size=None):
    """
    Generates the height-tracking PDDL actions.

    Height increases only when the relation between consecutive stack objects
    matches the NORMAL symbol (plain stacking).  If any pair matches an
    INSERTED symbol (on-top / inside), the height stays the same — check_inserted fires.

    Parameters
    ----------
    collapse_syms  : list of list[int]
    inserted_syms  : list of list[int]
    normal_sym     : list[int]
    sym_size       : int

    Returns
    -------
    str  — PDDL action definitions
    """

    actions = ""

    # ── count-1: always H0 → H1 (first object placed, no relations yet) ──────
    actions += "(:action increase_height_count_1\n"
    actions += "\t:parameters (?o0)\n"
    actions += "\t:precondition (and\n"
    actions += "\t\t(active-count-1)\n"
    actions += "\t\t(pending-height-check)\n"
    actions += "\t\t(H0)\n"
    actions += "\t\t(top-1 ?o0)\n"
    actions += "\t)\n"
    actions += "\t:effect (and\n"
    actions += "\t\t(not (H0)) (H1)\n"
    actions += "\t\t(not (pending-height-check))\n"
    actions += "\t)\n"
    actions += ")\n\n"

    # ── count-2: H1 → H2 only when top-2 vs top-1 matches the normal symbol ──
    normal_pair_2 = bits_to_pddl_pair(normal_sym, "?o0", "?o1")

    actions += "(:action increase_height_count_2\n"
    actions += "\t:parameters (?o0 ?o1)\n"
    actions += "\t:precondition (and\n"
    actions += "\t\t(active-count-2)\n"
    actions += "\t\t(pending-height-check)\n"
    actions += "\t\t(H1)\n"
    actions += "\t\t(top-2 ?o0) (top-1 ?o1)\n"
    actions += "\t\t(not (= ?o0 ?o1))\n"
    actions += f"\t\t{normal_pair_2}\n"   # ← uses actual normal symbol
    actions += "\t)\n"
    actions += "\t:effect (and\n"
    actions += "\t\t(not (H1)) (H2)\n"
    actions += "\t\t(not (pending-height-check))\n"
    actions += "\t)\n"
    actions += ")\n\n"

    # check_inserted_count_2: top pair matches an inserted symbol → no H increase
    actions += "(:action check_inserted_count_2\n"
    actions += "\t:parameters (?o0 ?o1)\n"
    actions += "\t:precondition (and\n"
    actions += "\t\t(active-count-2)\n"
    actions += "\t\t(pending-height-check)\n"
    actions += "\t\t(top-2 ?o0) (top-1 ?o1)\n"
    actions += "\t\t(not (= ?o0 ?o1))\n"
    if inserted_syms:
        actions += "\t\t(or\n"
        for isym in inserted_syms:
            actions += f"\t\t\t{bits_to_pddl_pair(isym, '?o0', '?o1')}\n"
        actions += "\t\t)\n"
    else:
        # No inserted symbols exist in this model — action is unreachable but
        # we keep it structurally valid with a tautology-like fallback.
        actions += "\t\t; NOTE: no inserted symbols found for this model\n"
        actions += f"\t\t{bits_to_pddl_pair(normal_sym, '?o0', '?o1')} ; unreachable\n"
    actions += "\t)\n"
    actions += "\t:effect (and\n"
    actions += "\t\t(not (pending-height-check))\n"
    actions += "\t)\n"
    actions += ")\n\n"

    # ── count-3 WITHOUT top-4: H1→H10 ────────────────────────────────────────
    # Height increases when BOTH (top-2 vs top-1) AND (top-3 vs top-1) are normal
    for h in range(1, 10):
        actions += f"(:action increase_height_h{h}_count_3\n"
        actions += f"\t:parameters (?o0 ?o1 ?o2)\n"
        actions += f"\t:precondition (and\n"
        actions += f"\t\t(active-count-3)\n"
        actions += f"\t\t(pending-height-check)\n"
        actions += f"\t\t(H{h})\n"
        actions += f"\t\t(not (has-top-4))\n"
        actions += f"\t\t(top-3 ?o0) (top-2 ?o1) (top-1 ?o2)\n"
        actions += f"\t\t(not (= ?o0 ?o1)) (not (= ?o0 ?o2)) (not (= ?o1 ?o2))\n"
        # Both pairs must match the normal symbol for a clean height increase
        actions += f"\t\t{bits_to_pddl_pair(normal_sym, '?o1', '?o2')}\n"  # top-2 vs top-1
        actions += f"\t\t{bits_to_pddl_pair(normal_sym, '?o0', '?o2')}\n"  # top-3 vs top-1
        actions += f"\t)\n"
        actions += f"\t:effect (and\n"
        actions += f"\t\t(not (H{h})) (H{h+1})\n"
        actions += f"\t\t(not (pending-height-check))\n"
        actions += f"\t)\n"
        actions += ")\n\n"

    # check_inserted_count_3 (no top-4): at least one pair is inserted
    actions += "(:action check_inserted_count_3\n"
    actions += "\t:parameters (?o0 ?o1 ?o2)\n"
    actions += "\t:precondition (and\n"
    actions += "\t\t(active-count-3)\n"
    actions += "\t\t(pending-height-check)\n"
    actions += "\t\t(not (has-top-4))\n"
    actions += "\t\t(top-3 ?o0) (top-2 ?o1) (top-1 ?o2)\n"
    actions += "\t\t(not (= ?o0 ?o1)) (not (= ?o0 ?o2)) (not (= ?o1 ?o2))\n"
    if inserted_syms:
        actions += "\t\t(or\n"
        for isym in inserted_syms:
            actions += f"\t\t\t{bits_to_pddl_pair(isym, '?o1', '?o2')}\n"  # top-2 vs top-1
            actions += f"\t\t\t{bits_to_pddl_pair(isym, '?o0', '?o2')}\n"  # top-3 vs top-1
        actions += "\t\t)\n"
    else:
        actions += "\t\t; NOTE: no inserted symbols found for this model\n"
        actions += f"\t\t{bits_to_pddl_pair(normal_sym, '?o1', '?o2')} ; unreachable\n"
    actions += "\t)\n"
    actions += "\t:effect (and\n"
    actions += "\t\t(not (pending-height-check))\n"
    actions += "\t)\n"
    actions += ")\n\n"

    # ── count-3 WITH top-4: H1→H10 ───────────────────────────────────────────
    for h in range(1, 10):
        actions += f"(:action increase_height_h{h}_count_3_top4\n"
        actions += f"\t:parameters (?o0 ?o1 ?o2 ?o3)\n"
        actions += f"\t:precondition (and\n"
        actions += f"\t\t(active-count-3)\n"
        actions += f"\t\t(pending-height-check)\n"
        actions += f"\t\t(H{h})\n"
        actions += f"\t\t(has-top-4)\n"
        actions += f"\t\t(top-4 ?o0) (top-3 ?o1) (top-2 ?o2) (top-1 ?o3)\n"
        actions += f"\t\t(not (= ?o0 ?o1)) (not (= ?o0 ?o2)) (not (= ?o0 ?o3))\n"
        actions += f"\t\t(not (= ?o1 ?o2)) (not (= ?o1 ?o3)) (not (= ?o2 ?o3))\n"
        actions += f"\t\t{bits_to_pddl_pair(normal_sym, '?o2', '?o3')}\n"  # top-2 vs top-1
        actions += f"\t\t{bits_to_pddl_pair(normal_sym, '?o1', '?o3')}\n"  # top-3 vs top-1
        actions += f"\t\t{bits_to_pddl_pair(normal_sym, '?o0', '?o3')}\n"  # top-4 vs top-1
        actions += f"\t)\n"
        actions += f"\t:effect (and\n"
        actions += f"\t\t(not (H{h})) (H{h+1})\n"
        actions += f"\t\t(not (pending-height-check))\n"
        actions += f"\t)\n"
        actions += ")\n\n"

    # check_inserted_count_3_top4: at least one pair is inserted
    actions += "(:action check_inserted_count_3_top4\n"
    actions += "\t:parameters (?o0 ?o1 ?o2 ?o3)\n"
    actions += "\t:precondition (and\n"
    actions += "\t\t(active-count-3)\n"
    actions += "\t\t(pending-height-check)\n"
    actions += "\t\t(has-top-4)\n"
    actions += "\t\t(top-4 ?o0) (top-3 ?o1) (top-2 ?o2) (top-1 ?o3)\n"
    actions += "\t\t(not (= ?o0 ?o1)) (not (= ?o0 ?o2)) (not (= ?o0 ?o3))\n"
    actions += "\t\t(not (= ?o1 ?o2)) (not (= ?o1 ?o3)) (not (= ?o2 ?o3))\n"
    if inserted_syms:
        actions += "\t\t(or\n"
        for isym in inserted_syms:
            actions += f"\t\t\t{bits_to_pddl_pair(isym, '?o2', '?o3')}\n"  # top-2 vs top-1
            actions += f"\t\t\t{bits_to_pddl_pair(isym, '?o1', '?o3')}\n"  # top-3 vs top-1
            actions += f"\t\t\t{bits_to_pddl_pair(isym, '?o0', '?o3')}\n"  # top-4 vs top-1
        actions += "\t\t)\n"
    else:
        actions += "\t\t; NOTE: no inserted symbols found for this model\n"
        actions += f"\t\t{bits_to_pddl_pair(normal_sym, '?o2', '?o3')} ; unreachable\n"
    actions += "\t)\n"
    actions += "\t:effect (and\n"
    actions += "\t\t(not (pending-height-check))\n"
    actions += "\t)\n"
    actions += ")\n\n"

    return actions