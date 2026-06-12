"""
extend_towers_pool_based.py
──────────────────────────────────────────────────────────────────────────────
Pool-based tower experiments.

This script DOES NOT globally extend every height-4 tower.

Instead, for each target height H:
  1. Randomly selects N different object pools/multisets of length H.
  2. Each selected pool must have at least one valid ordering.
  3. For every selected pool, generates ALL UNIQUE VALID PERMUTATIONS.
  4. A permutation is valid if every sliding window of the last 4 objects obeys:
       - tuple limits
       - tuple pair/mutual-exclusion limits
  5. Simulates every valid permutation from scratch in PyBullet.
  6. Saves:
       - height_H.csv
       - height_H_done.txt
       - height_H_pools.csv
       - summary.csv

Example:
    python extend_towers_pool_based.py \
        --out_dir pool_based_height_csv \
        --min_height 5 \
        --max_height 8 \
        --pools_per_height 20 \
        --workers 8 \
        --seed 0

Debug example:
    python extend_towers_pool_based.py \
        --out_dir pool_based_debug \
        --min_height 5 \
        --max_height 5 \
        --pools_per_height 2 \
        --max_sequences_per_pool 10 \
        --workers 1
"""

import os
import csv
import uuid
import time
import argparse
import random
import multiprocessing as mp
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
import pybullet as p

# Keep compatibility with your original folder structure.
try:
    from data_collection_direct.env import PyBulletEnvironment
except ImportError:
    from env import PyBulletEnvironment


# ─────────────────────────────────────────────────────────────────────────────
# Tuple rules
# ─────────────────────────────────────────────────────────────────────────────

TUPLE_LIMITS = {
    ("inverted_cup", "inv_cup_w20_h20_thinner.urdf"): 1,
    ("inverted_cup", "inv_cup_w18_h18_thinner.urdf"): 1,
    ("inverted_cup", "inv_cup_w16_h16_thinner.urdf"): 1,
    ("cup",          "cup_w20_h20.urdf"):              1,
    ("cup",          "cup_w18_h18.urdf"):              1,
    ("cup",          "cup_w16_h16.urdf"):              1,
    ("sphere",       "0.14"):                          5,
    ("box",          "0.14"):                          1,
}

TUPLE_PAIRS = [
    (("cup", "cup_w20_h20.urdf"), ("inverted_cup", "inv_cup_w20_h20_thinner.urdf")),
    (("cup", "cup_w18_h18.urdf"), ("inverted_cup", "inv_cup_w18_h18_thinner.urdf")),
    (("cup", "cup_w16_h16.urdf"), ("inverted_cup", "inv_cup_w16_h16_thinner.urdf")),
]

DEFAULT_OBJECT_POOL = [
    ("inverted_cup", "inv_cup_w20_h20_thinner.urdf"),
    ("inverted_cup", "inv_cup_w18_h18_thinner.urdf"),
    ("inverted_cup", "inv_cup_w16_h16_thinner.urdf"),
    ("cup",          "cup_w20_h20.urdf"),
    ("cup",          "cup_w18_h18.urdf"),
    ("cup",          "cup_w16_h16.urdf"),
    ("sphere",       "0.14"),
    ("box",          "0.14"),
]


CSV_COLUMNS = [
    "object_type", "object_size_or_path", "position",
    "collapse", "collapse_type", "collapse_object_type",
    "global_collapse", "pairwise_collapse", "reasons",
    "id", "step",
    "bounding_box_differences", "bbox", "overlap",
    "occluded_objects", "spatial_relations",
    "rgb_image_path", "depth_image_path",
    "upper_image_path", "upper_depth_image_path",
    "lower_image_path", "lower_depth_image_path",

    # New metadata for pool-based generation
    "height",
    "pool_id",
    "pool_key",
    "permutation_index",
    "sequence_key",
    "continuous_height",
]


_WRITER_STOP = "__STOP__"


# ─────────────────────────────────────────────────────────────────────────────
# Canonical key helpers
# ─────────────────────────────────────────────────────────────────────────────

def _canonical_size(raw: str) -> str:
    """
    Normalize a size/path string to a stable short key.
      - URDF path/name -> basename without extension
      - float string   -> rounded to 6 dp, trailing zeros stripped
    """
    raw = str(raw).strip()

    if "/" in raw or raw.lower().endswith(".urdf"):
        return os.path.splitext(os.path.basename(raw))[0]

    try:
        v = float(raw)
        return f"{v:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return raw


def tuple_key(obj) -> tuple:
    """Canonical (type, size/path) key."""
    obj_type, raw = obj
    return (str(obj_type), _canonical_size(raw))


def sequence_key(seq) -> str:
    """Canonical ordered key for one full tower sequence."""
    return "|".join(f"{t}:{s}" for t, s in (tuple_key(o) for o in seq))


def pool_key(pool) -> str:
    """
    Canonical unordered/multiset key for an object pool.
    Same objects in different order produce the same key.
    """
    keys = [f"{t}:{s}" for t, s in (tuple_key(o) for o in pool)]
    keys.sort()
    return "|".join(keys)


# ─────────────────────────────────────────────────────────────────────────────
# Pair lookup and canonical tuple limits
# ─────────────────────────────────────────────────────────────────────────────

def _build_pair_lookup():
    lookup = {}
    for a, b in TUPLE_PAIRS:
        ka, kb = tuple_key(a), tuple_key(b)
        lookup[ka] = kb
        lookup[kb] = ka
    return lookup


PAIR_LOOKUP = _build_pair_lookup()
TUPLE_LIMITS_CANONICAL = {tuple_key(k): v for k, v in TUPLE_LIMITS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Tuple/window validation
# ─────────────────────────────────────────────────────────────────────────────

def window_valid(window) -> bool:
    """
    Check tuple limits and tuple-pair limits inside one window.
    This is the important local rule.
    """
    counts = defaultdict(int)
    for obj in window:
        counts[tuple_key(obj)] += 1

    # Individual tuple limits
    for key, count in counts.items():
        limit = TUPLE_LIMITS_CANONICAL.get(key, float("inf"))
        if count > limit:
            return False

    # Pair/mutual-exclusion limits
    seen_pairs = set()
    for key in list(counts):
        partner = PAIR_LOOKUP.get(key)
        if partner is None:
            continue

        pair = tuple(sorted([key, partner]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        combined = counts.get(key, 0) + counts.get(partner, 0)

        limit_a = TUPLE_LIMITS_CANONICAL.get(key, float("inf"))
        limit_b = TUPLE_LIMITS_CANONICAL.get(partner, float("inf"))

        if combined > min(limit_a, limit_b):
            return False

    return True


def all_sliding_windows_valid(seq) -> bool:
    """
    For height > 4, check every last-4 sliding window.
    For height <= 4, check the whole sequence.
    """
    if len(seq) <= 4:
        return window_valid(seq)

    for i in range(len(seq) - 3):
        if not window_valid(seq[i:i + 4]):
            return False

    return True


def prefix_valid_last4(prefix) -> bool:
    """
    During permutation backtracking, only the newest last-4 window can become invalid.
    """
    if len(prefix) <= 4:
        return window_valid(prefix)
    return window_valid(prefix[-4:])


# ─────────────────────────────────────────────────────────────────────────────
# Done-set
# ─────────────────────────────────────────────────────────────────────────────

def done_set_path(out_dir: str, height: int) -> str:
    return os.path.join(out_dir, f"height_{height}_done.txt")


def load_done_set(out_dir: str, height: int) -> set:
    path = done_set_path(out_dir, height)
    if not os.path.exists(path):
        return set()

    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


# ─────────────────────────────────────────────────────────────────────────────
# Object pool construction
# ─────────────────────────────────────────────────────────────────────────────

def build_object_pool(object_dir: str) -> list:
    """
    Build the base object choices used for sampling pools.
    Cups/inverted cups get object_dir prepended.
    Sphere/box use numeric size.
    """
    pool = []
    for obj_type, raw in DEFAULT_OBJECT_POOL:
        if obj_type in ("cup", "inverted_cup", "ring"):
            pool.append((obj_type, os.path.join(object_dir, raw)))
        else:
            pool.append((obj_type, raw))
    return pool


# ─────────────────────────────────────────────────────────────────────────────
# Permutation generation for each object pool
# ─────────────────────────────────────────────────────────────────────────────

def has_at_least_one_valid_permutation(pool) -> bool:
    """
    Fast acceptance test for sampled pools.
    Stops immediately when it finds one valid ordering.
    """
    counter = Counter(tuple_key(o) for o in pool)
    key_to_obj = {tuple_key(o): o for o in pool}
    n = len(pool)

    def backtrack(prefix):
        if len(prefix) == n:
            return True

        for key in list(counter.keys()):
            if counter[key] <= 0:
                continue

            obj = key_to_obj[key]
            prefix.append(obj)
            counter[key] -= 1

            ok = False
            if prefix_valid_last4(prefix):
                ok = backtrack(prefix)

            counter[key] += 1
            prefix.pop()

            if ok:
                return True

        return False

    return backtrack([])


def valid_permutations_for_pool(pool, max_sequences_per_pool=0):
    """
    Generate ALL unique valid permutations/orderings for one object pool.

    Repeated objects are handled correctly because we backtrack over a Counter.

    max_sequences_per_pool:
      - 0 means no cap; generate all valid permutations.
      - positive value is only for debugging.
    """
    counter = Counter(tuple_key(o) for o in pool)
    key_to_obj = {tuple_key(o): o for o in pool}
    n = len(pool)
    results = []

    def backtrack(prefix):
        if max_sequences_per_pool > 0 and len(results) >= max_sequences_per_pool:
            return

        if len(prefix) == n:
            results.append(tuple(prefix))
            return

        for key in list(counter.keys()):
            if counter[key] <= 0:
                continue

            obj = key_to_obj[key]

            prefix.append(obj)
            counter[key] -= 1

            if prefix_valid_last4(prefix):
                backtrack(prefix)

            counter[key] += 1
            prefix.pop()

    backtrack([])
    return results


def sample_one_pool(height, object_pool, rng):
    """
    Randomly sample one unordered object pool/multiset of length `height`.

    The full pool itself is allowed to have repeated objects as long as
    there exists at least one valid ordering under the sliding last-4 rule.
    """
    return tuple(rng.choice(object_pool) for _ in range(height))


def sample_object_pools_for_height(
    height,
    object_pool,
    n_pools,
    rng,
    max_sampling_attempts=100000,
    banned_pool_keys=None,
):
    """
    Select n_pools different valid object pools for a target height.

    A pool is accepted only if:
      - it is not already in banned_pool_keys
      - it has at least one valid permutation

    In --continue mode, banned_pool_keys comes from the existing
    height_H_pools.csv file, so new runs add different object pools.
    """
    banned_pool_keys = set(banned_pool_keys or [])

    selected = []
    seen_pool_keys = set()
    attempts = 0

    while len(selected) < n_pools and attempts < max_sampling_attempts:
        attempts += 1

        pool = sample_one_pool(height, object_pool, rng)
        pk = pool_key(pool)

        if pk in banned_pool_keys:
            continue

        if pk in seen_pool_keys:
            continue

        if not has_at_least_one_valid_permutation(pool):
            continue

        seen_pool_keys.add(pk)
        selected.append(pool)

    if len(selected) < n_pools:
        print(
            f"WARNING: only found {len(selected)} valid new pools for height {height} "
            f"after {attempts} attempts. Existing/banned pools skipped: "
            f"{len(banned_pool_keys)}."
        )

    return selected


def generate_pool_based_candidates(
    height,
    object_pool,
    n_pools,
    done_keys,
    rng,
    max_sequences_per_pool=0,
    banned_pool_keys=None,
    pool_start_index=0,
    max_sampling_attempts=100000,
):
    """
    For one height:
      1. sample/select n_pools valid object pools
      2. generate ALL valid permutations for each pool
      3. skip sequences already in done_keys
      4. return simulation candidates + pool summary rows

    In --continue mode:
      - banned_pool_keys prevents reusing existing object pools
      - pool_start_index prevents pool_id collisions
    """
    pools = sample_object_pools_for_height(
        height=height,
        object_pool=object_pool,
        n_pools=n_pools,
        rng=rng,
        max_sampling_attempts=max_sampling_attempts,
        banned_pool_keys=banned_pool_keys,
    )

    candidates = []
    pool_summaries = []

    for local_pool_idx, pool in enumerate(pools):
        pool_idx = pool_start_index + local_pool_idx
        pool_id = f"H{height}_pool{pool_idx:03d}"
        pk = pool_key(pool)

        valid_orders = valid_permutations_for_pool(
            pool,
            max_sequences_per_pool=max_sequences_per_pool,
        )

        n_all_valid = len(valid_orders)
        n_added = 0

        for perm_idx, seq in enumerate(valid_orders):
            key = sequence_key(seq)

            if key in done_keys:
                continue

            done_keys.add(key)

            candidates.append({
                "height": height,
                "pool_id": pool_id,
                "pool_key": pk,
                "permutation_index": perm_idx,
                "sequence": seq,
                "_key": key,
            })
            n_added += 1

        pool_summaries.append({
            "height": height,
            "pool_id": pool_id,
            "pool_key": pk,
            "pool_objects": str(list(pool)),
            "all_valid_permutations": n_all_valid,
            "new_after_done_filter": n_added,
        })

        print(
            f"  {pool_id}: valid permutations={n_all_valid}, "
            f"new to simulate={n_added}"
        )

    return candidates, pool_summaries


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_size(raw: str, object_dir: str, obj_type: str) -> str:
    """
    Convert relative URDF filenames to full paths for PyBullet.
    Numeric geometric sizes stay numeric strings.
    """
    raw = str(raw)

    if obj_type in ("sphere", "box", "cylinder"):
        try:
            v = float(raw)
            return f"{v:.6f}".rstrip("0").rstrip(".")
        except ValueError:
            return raw

    if os.path.isabs(raw) or os.path.dirname(raw):
        return raw

    return os.path.join(object_dir, raw)


def _safe(x) -> str:
    try:
        return str(x)
    except Exception:
        return "None"


def _continuous_height(bbox: dict):
    zmins, zmaxs = [], []

    for v in bbox.values():
        try:
            zmins.append(float(v["min"][2]))
            zmaxs.append(float(v["max"][2]))
        except Exception:
            pass

    if not zmins:
        return None

    return max(zmaxs) - min(zmins)


def _error_row(
    raw_obj,
    tower_uid,
    step_i,
    height,
    pool_id,
    pk,
    permutation_index,
    seq_key,
    error,
):
    obj_type, raw = raw_obj

    return {
        "object_type": obj_type,
        "object_size_or_path": raw,
        "position": "None",

        "collapse": True,
        "collapse_type": error,
        "collapse_object_type": obj_type,
        "global_collapse": True,
        "pairwise_collapse": "{}",
        "reasons": error,

        "id": tower_uid,
        "step": step_i,

        "bounding_box_differences": "None",
        "bbox": "{}",
        "overlap": "None",
        "occluded_objects": "{}",
        "spatial_relations": "{}",

        "rgb_image_path": "NOT_SAVED",
        "depth_image_path": "NOT_SAVED",
        "upper_image_path": "NOT_SAVED",
        "upper_depth_image_path": "NOT_SAVED",
        "lower_image_path": "NOT_SAVED",
        "lower_depth_image_path": "NOT_SAVED",

        "height": height,
        "pool_id": pool_id,
        "pool_key": pk,
        "permutation_index": permutation_index,
        "sequence_key": seq_key,
        "continuous_height": None,
    }


def simulate_sequence(
    seq,
    height,
    pool_id,
    pk,
    permutation_index,
    seq_key,
    object_dir,
    sim_steps,
    settle_steps,
):
    """
    Simulate a full tower sequence from scratch.
    Returns (rows, noncollapsed_bool).

    Note:
      Images are not saved in this efficient CSV version.
    """
    env = PyBulletEnvironment()
    tower_uid = uuid.uuid4().hex[:12]
    rows = []

    try:
        for step_i, raw_obj in enumerate(seq, start=1):
            obj_type = raw_obj[0]
            obj_size = _normalise_size(raw_obj[1], object_dir, obj_type)

            try:
                obj_id, actual_type, actual_size, position = env.choose_object_and_place(
                    object_type=obj_type,
                    object_size_or_path=obj_size,
                    position=[0, 0, 0],
                )
            except Exception as ex:
                rows.append(
                    _error_row(
                        raw_obj, tower_uid, step_i,
                        height, pool_id, pk, permutation_index, seq_key,
                        f"spawn_error: {ex}",
                    )
                )
                return rows, False

            try:
                env.place_object_on_stack(obj_id, position)
            except Exception as ex:
                rows.append(
                    _error_row(
                        raw_obj, tower_uid, step_i,
                        height, pool_id, pk, permutation_index, seq_key,
                        f"place_error: {ex}",
                    )
                )
                return rows, False

            # Simulate and settle.
            for _ in range(sim_steps):
                p.stepSimulation()

            for _ in range(settle_steps):
                p.stepSimulation()
                try:
                    all_still = all(
                        max(abs(v) for v in
                            p.getBaseVelocity(oid)[0] +
                            p.getBaseVelocity(oid)[1]) < 1e-3
                        for oid in env.placed_objects
                    )
                    if all_still:
                        break
                except Exception:
                    pass

            for _ in range(sim_steps):
                p.stepSimulation()

            # Collapse / relations / bbox
            collapse_type, collapse, collapse_obj_type = env.detect_collapse_new()

            try:
                pairwise_res = env.detect_collapse_pairwise()
            except Exception as ex:
                pairwise_res = {
                    "global_collapse": collapse,
                    "object_status": {},
                    "reasons": {"pairwise_error": str(ex)},
                }

            global_col = pairwise_res.get("global_collapse", collapse)

            try:
                spatial = env.get_all_spatial_relations()
            except Exception:
                spatial = {}

            try:
                _, _, bbox_diffs, overlap = env.calc_bbox_diff_sign()
            except Exception:
                bbox_diffs, overlap = None, None

            try:
                all_bbox = env.get_all_placed_aabbs()
            except Exception:
                all_bbox = {}

            rows.append({
                "object_type": actual_type,
                "object_size_or_path": actual_size,
                "position": _safe(position),

                "collapse": bool(collapse),
                "collapse_type": collapse_type,
                "collapse_object_type": collapse_obj_type,
                "global_collapse": bool(global_col),
                "pairwise_collapse": _safe(pairwise_res.get("object_status", {})),
                "reasons": _safe(pairwise_res.get("reasons", {})),

                "id": tower_uid,
                "step": step_i,

                "bounding_box_differences": _safe(bbox_diffs),
                "bbox": _safe(all_bbox),
                "overlap": _safe(overlap),
                "occluded_objects": _safe(spatial.get("occluding", {})),
                "spatial_relations": _safe(spatial),

                "rgb_image_path": "NOT_SAVED",
                "depth_image_path": "NOT_SAVED",
                "upper_image_path": "NOT_SAVED",
                "upper_depth_image_path": "NOT_SAVED",
                "lower_image_path": "NOT_SAVED",
                "lower_depth_image_path": "NOT_SAVED",

                "height": height,
                "pool_id": pool_id,
                "pool_key": pk,
                "permutation_index": permutation_index,
                "sequence_key": seq_key,
                "continuous_height": _continuous_height(all_bbox),
            })

            if collapse or global_col:
                return rows, False

        return rows, True

    finally:
        try:
            env.reset()
        except Exception:
            pass

        try:
            p.disconnect()
        except Exception:
            pass


def _worker(args):
    (
        seq,
        height,
        pool_id,
        pk,
        permutation_index,
        seq_key,
        object_dir,
        sim_steps,
        settle_steps,
    ) = args

    rows, ok = simulate_sequence(
        seq=seq,
        height=height,
        pool_id=pool_id,
        pk=pk,
        permutation_index=permutation_index,
        seq_key=seq_key,
        object_dir=object_dir,
        sim_steps=sim_steps,
        settle_steps=settle_steps,
    )

    return seq, pool_id, permutation_index, seq_key, rows, ok


# ─────────────────────────────────────────────────────────────────────────────
# Writer process
# ─────────────────────────────────────────────────────────────────────────────

def _writer_process(queue: mp.Queue, csv_path: str, done_file: str, batch_size: int):
    """
    Drains the result queue and writes rows to CSV in batches.
    Also appends sequence keys to the done-set file.

    Fixed:
      - uses == for sentinel
      - does not accidentally overwrite previous flushes
    """
    write_header = not os.path.exists(csv_path)
    pending_rows = []
    pending_keys = []

    def flush():
        nonlocal write_header

        if not pending_rows:
            return

        mode = "w" if write_header else "a"

        with open(csv_path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerows(pending_rows)

        with open(done_file, "a") as f:
            for k in pending_keys:
                f.write(k + "\n")

        pending_rows.clear()
        pending_keys.clear()
        write_header = False

    while True:
        item = queue.get()

        if item == _WRITER_STOP:
            flush()
            break

        _seq, _pool_id, _perm_idx, seq_key, rows, _ok = item

        pending_rows.extend(rows)
        pending_keys.append(seq_key)

        if len(pending_rows) >= batch_size:
            flush()


# ─────────────────────────────────────────────────────────────────────────────
# Continue-mode helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_pool_keys(pool_summary_csv: str) -> set:
    """
    Read existing height_H_pools.csv and return pool_key values.
    Used by --continue so newly sampled pools are different from existing ones.
    """
    if not os.path.exists(pool_summary_csv):
        return set()

    try:
        df = pd.read_csv(pool_summary_csv)
    except Exception as ex:
        print(f"WARNING: could not read existing pool summary {pool_summary_csv}: {ex}")
        return set()

    if "pool_key" not in df.columns:
        return set()

    return set(df["pool_key"].dropna().astype(str).tolist())


def next_pool_start_index(pool_summary_csv: str, height: int) -> int:
    """
    Find the next available pool index for pool ids like H5_pool000.
    If the summary file is missing, returns 0.
    """
    if not os.path.exists(pool_summary_csv):
        return 0

    try:
        df = pd.read_csv(pool_summary_csv)
    except Exception:
        return 0

    if "pool_id" not in df.columns:
        return len(df)

    prefix = f"H{height}_pool"
    max_idx = -1

    for raw in df["pool_id"].dropna().astype(str):
        if not raw.startswith(prefix):
            continue
        try:
            idx = int(raw.replace(prefix, ""))
            max_idx = max(max_idx, idx)
        except ValueError:
            pass

    if max_idx >= 0:
        return max_idx + 1

    return len(df)


def save_pool_summaries(pool_summary_csv: str, pool_summaries: list, append: bool):
    """
    Save per-height pool summaries.
    In --continue mode, append new pools to the existing CSV.
    """
    df = pd.DataFrame(pool_summaries)

    if append and os.path.exists(pool_summary_csv):
        df.to_csv(pool_summary_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(pool_summary_csv, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", default="pool_based_height_csv")
    parser.add_argument("--object_dir", default="objects_symcan")

    parser.add_argument("--min_height", type=int, default=5)
    parser.add_argument("--max_height", type=int, default=8)

    parser.add_argument("--pools_per_height", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--sim_steps", type=int, default=1200)
    parser.add_argument("--settle_steps", type=int, default=200)
    parser.add_argument("--write_batch", type=int, default=200)

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for selecting object pools. "
            "Different seeds give different pool combinations. "
            "If omitted, a fresh seed is generated and printed."
        ),
    )

    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help=(
            "Continue an existing out_dir by adding new object pools that are "
            "different from the pools already saved in height_H_pools.csv."
        ),
    )

    parser.add_argument(
        "-n",
        "--additional_pools",
        type=int,
        default=None,
        help=(
            "Number of additional object pools per height to add in --continue mode. "
            "Example: --continue -n 10 adds 10 new pools per height."
        ),
    )

    parser.add_argument(
        "--max_sequences_per_pool",
        type=int,
        default=0,
        help=(
            "0 = generate and simulate ALL valid permutations. "
            "Positive value = debug cap per pool."
        ),
    )

    parser.add_argument(
        "--max_sampling_attempts",
        type=int,
        default=100000,
        help="Maximum attempts for finding valid object pools per height.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.continue_run and args.additional_pools is None:
        args.additional_pools = args.pools_per_height

    if not args.continue_run and args.additional_pools is not None:
        print("NOTE: --additional_pools/-n is only used with --continue.")

    if args.seed is None:
        args.seed = time.time_ns() % (2**32)

    rng = random.Random(args.seed)

    print("=" * 80)
    print("POOL-BASED TOWER EXPERIMENT")
    print("=" * 80)
    for k, v in vars(args).items():
        print(f"  {k:<30}: {v}")
    print("=" * 80)

    object_pool = build_object_pool(args.object_dir)

    print("\nBase object choices:")
    for obj in object_pool:
        print(f"  {obj}")

    summary_rows = []

    for target_height in range(args.min_height, args.max_height + 1):
        print("\n" + "#" * 80)
        print(f"# HEIGHT {target_height}")
        print("#" * 80)

        out_csv = os.path.join(args.out_dir, f"height_{target_height}.csv")
        done_file = done_set_path(args.out_dir, target_height)
        done_keys = load_done_set(args.out_dir, target_height)

        print(f"  Already done from previous runs: {len(done_keys)} sequences")

        pool_summary_csv = os.path.join(
            args.out_dir,
            f"height_{target_height}_pools.csv",
        )

        existing_pool_keys = load_existing_pool_keys(pool_summary_csv) if args.continue_run else set()
        pool_start_index = next_pool_start_index(pool_summary_csv, target_height) if args.continue_run else 0
        n_pools_for_this_run = args.additional_pools if args.continue_run else args.pools_per_height

        if args.continue_run:
            print(f"  Continue mode: existing pools for H{target_height}: {len(existing_pool_keys)}")
            print(f"  Continue mode: adding new pools: {n_pools_for_this_run}")
            print(f"  Continue mode: next pool index starts at: {pool_start_index}")

        candidates, pool_summaries = generate_pool_based_candidates(
            height=target_height,
            object_pool=object_pool,
            n_pools=n_pools_for_this_run,
            done_keys=done_keys,
            rng=rng,
            max_sequences_per_pool=args.max_sequences_per_pool,
            banned_pool_keys=existing_pool_keys,
            pool_start_index=pool_start_index,
            max_sampling_attempts=args.max_sampling_attempts,
        )

        save_pool_summaries(
            pool_summary_csv,
            pool_summaries,
            append=args.continue_run,
        )

        print(f"\n  Selected new pools: {len(pool_summaries)}")
        print(f"  New sequences to simulate: {len(candidates)}")
        print(f"  Pool summary saved to: {pool_summary_csv}")

        if not candidates:
            print("  No new candidates for this height.")
            summary_rows.append({
                "height": target_height,
                "mode": "continue" if args.continue_run else "new",
                "seed": args.seed,
                "pools": len(pool_summaries),
                "candidates": 0,
                "noncollapsed": 0,
                "collapsed_or_error": 0,
                "elapsed_sec": 0,
                "height_csv": out_csv,
                "pool_summary_csv": pool_summary_csv,
            })
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(args.out_dir, "summary.csv"),
                index=False,
            )
            continue

        result_queue = mp.Queue(maxsize=max(4, args.workers * 4))

        writer = mp.Process(
            target=_writer_process,
            args=(result_queue, out_csv, done_file, args.write_batch),
            daemon=True,
        )
        writer.start()

        worker_args = [
            (
                item["sequence"],
                item["height"],
                item["pool_id"],
                item["pool_key"],
                item["permutation_index"],
                item["_key"],
                args.object_dir,
                args.sim_steps,
                args.settle_steps,
            )
            for item in candidates
        ]

        n_done = 0
        n_ok = 0
        n_bad = 0
        t0 = time.time()

        if args.workers <= 1:
            for wa in worker_args:
                seq, pool_id, perm_idx, key, rows, ok = _worker(wa)
                result_queue.put((seq, pool_id, perm_idx, key, rows, ok))

                n_done += 1
                if ok:
                    n_ok += 1
                else:
                    n_bad += 1

                if n_done % 50 == 0 or n_done == len(candidates):
                    print(
                        f"  done={n_done}/{len(candidates)} "
                        f"ok={n_ok} bad={n_bad}"
                    )
        else:
            with mp.Pool(processes=args.workers) as pool:
                for seq, pool_id, perm_idx, key, rows, ok in pool.imap_unordered(
                    _worker,
                    worker_args,
                    chunksize=1,
                ):
                    result_queue.put((seq, pool_id, perm_idx, key, rows, ok))

                    n_done += 1
                    if ok:
                        n_ok += 1
                    else:
                        n_bad += 1

                    if n_done % 50 == 0 or n_done == len(candidates):
                        print(
                            f"  done={n_done}/{len(candidates)} "
                            f"ok={n_ok} bad={n_bad}"
                        )

        result_queue.put(_WRITER_STOP)
        writer.join()

        elapsed = time.time() - t0

        print(f"\n  HEIGHT {target_height} DONE")
        print(f"  selected pools   : {len(pool_summaries)}")
        print(f"  simulated        : {len(candidates)}")
        print(f"  non-collapsed    : {n_ok}")
        print(f"  collapsed/error  : {n_bad}")
        print(f"  csv              : {out_csv}")
        print(f"  elapsed          : {elapsed:.1f}s")

        summary_rows.append({
            "height": target_height,
            "mode": "continue" if args.continue_run else "new",
            "seed": args.seed,
            "pools": len(pool_summaries),
            "candidates": len(candidates),
            "noncollapsed": n_ok,
            "collapsed_or_error": n_bad,
            "elapsed_sec": round(elapsed, 3),
            "height_csv": out_csv,
            "pool_summary_csv": pool_summary_csv,
        })

        pd.DataFrame(summary_rows).to_csv(
            os.path.join(args.out_dir, "summary.csv"),
            index=False,
        )

    print("\nDONE")
    print(f"Summary: {os.path.join(args.out_dir, 'summary.csv')}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
