import os
import yaml
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
import pandas as pd

from model import load_ckpt
from load_data import load_data, preprocess_to_dataset_normalized, set_seed, calculate_bbox_stats
import load_data as _load_data
from utils import build_bank_optimized

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", required=True)
parser.add_argument("--tag",    default="best")
parser.add_argument("--split",  default="all",
                    choices=["train", "val", "test", "all"],
                    help="Which data split to analyse")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)
_load_data.init(cfg)

RUN_NAME         = cfg["run"]["name"]
SEED             = cfg["seed"]
TRAIN_VAL_SCRIPT = cfg["data"]["csv_path"]
UNIQUE_DEPTH_DIR = cfg["data"]["depth_image_dir"]
BATCH_SIZE       = cfg["training"]["batch_size"]

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_dir = os.path.join("logs", RUN_NAME)

# ── Load model ────────────────────────────────────────────────────────────────
model, _ = load_ckpt(RUN_NAME, tag=args.tag, device=device)
assert model.cluster_centroids.numel() > 0, \
    f"No centroids found for tag='{args.tag}'. Run fit_clusters.py first."

K             = model.cluster_centroids.shape[0]
obj_sym_size  = model.obj_symbol_size

# ── Data ──────────────────────────────────────────────────────────────────────
train_rows, val_rows, test_rows = load_data(TRAIN_VAL_SCRIPT)
STATS_FOR_NORM = calculate_bbox_stats(train_rows)

if args.split == "train":
    rows = train_rows
elif args.split == "val":
    rows = val_rows
elif args.split == "test":
    rows = test_rows
else:
    rows = train_rows + val_rows + test_rows

print(f"→ Analysing {len(rows)} rows ({args.split} split)")

imgs    = build_bank_optimized(rows, UNIQUE_DEPTH_DIR)
dataset = preprocess_to_dataset_normalized(rows, imgs, device, STATS_FOR_NORM)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Collect features + labels ─────────────────────────────────────────────────
print("→ Collecting features and cluster assignments...")
all_feats       = []
all_bits        = []
all_identities  = []   # "obj_type_clean_size"
all_obj_types   = []
all_obj_sizes   = []

model.eval()
with torch.no_grad():
    for batch in loader:
        batch.x = batch.x.float().to(device)

        # ONE call for the whole batch — correct BN statistics
        bits_batch, feats_batch = model.get_object_symbols(batch.x)
        # feats_batch = [Total_N, 64], bits_batch = [Total_N, obj_sym]

        for i in range(batch.num_graphs):
            node_ids  = (batch.batch == i).nonzero(as_tuple=True)[0]
            last_node = node_ids[-1]

            all_feats.append(feats_batch[last_node].unsqueeze(0).cpu())
            all_bits.append(bits_batch[last_node].unsqueeze(0).cpu())

            obj_type, obj_size = batch.meta_obj_type[i][0]
            clean_size = str(obj_size).split('/')[-1].replace('.urdf', '')
            identity   = f"{obj_type}_{clean_size}"
            all_identities.append(identity)
            all_obj_types.append(obj_type)
            all_obj_sizes.append(clean_size)

feats_all = torch.cat(all_feats, dim=0).numpy()   # [N, 64]
bits_all  = torch.cat(all_bits,  dim=0).numpy()   # [N, sym_size]

# Convert binary bits to cluster index
cluster_ids = np.array([
    int("".join(map(str, b.astype(int))), 2) for b in bits_all
])

# ── Per-cluster identity breakdown ───────────────────────────────────────────
print("\n" + "="*70)
print(f"{'CLUSTER ASSIGNMENT BREAKDOWN':^70}")
print("="*70)

cluster_to_identities = defaultdict(list)
for cid, ident in zip(cluster_ids, all_identities):
    cluster_to_identities[cid].append(ident)

rows_report = []
for cid in range(K):
    items    = cluster_to_identities[cid]
    counts   = defaultdict(int)
    for ident in items:
        counts[ident] += 1
    dominant = max(counts, key=counts.get) if counts else "—"
    purity   = counts[dominant] / len(items) if items else 0.0
    code     = format(cid, f'0{obj_sym_size}b')
    rows_report.append({
        "Cluster": cid,
        "Code":    code,
        "Size":    len(items),
        "Dominant": dominant,
        "Purity":  f"{purity:.1%}",
        **{k: counts.get(k, 0) for k in sorted(set(all_identities))}
    })
    if len(items) > 0:
        breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
        print(f"  Cluster {cid:2d} [{code}]  size={len(items):6d}  purity={purity:.1%}  | {breakdown}")
    else:
        print(f"  Cluster {cid:2d} [{code}]  size=0  ← EMPTY")

df_report = pd.DataFrame(rows_report)
report_path = os.path.join(log_dir, f"cluster_report_{args.tag}.csv")
df_report.to_csv(report_path, index=False)
print(f"\n→ Report saved to {report_path}")

# ── Metrics ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'CLUSTER QUALITY METRICS':^70}")
print("="*70)

# Map identity strings to integer labels for ARI
unique_identities = sorted(set(all_identities))
identity_to_int   = {ident: i for i, ident in enumerate(unique_identities)}
true_labels       = np.array([identity_to_int[i] for i in all_identities])

n_active = sum(1 for cid in range(K) if len(cluster_to_identities[cid]) > 0)
print(f"  Active clusters     : {n_active} / {K}")
print(f"  Unique object types : {len(unique_identities)}")

if n_active > 1:
    sil = silhouette_score(feats_all, cluster_ids, sample_size=min(5000, len(feats_all)))
    ari = adjusted_rand_score(true_labels, cluster_ids)
    print(f"  Silhouette score    : {sil:.4f}  (higher=better, max=1.0)")
    print(f"  Adjusted Rand Index : {ari:.4f}  (1.0=perfect match with ground truth)")
    print(f"\n  Interpretation:")
    if sil < 0.1:
        print("  ⚠️  Very low silhouette — clusters heavily overlap in feature space.")
        print("     Suggestion: reduce K to match number of unique objects, or")
        print("     improve image encoder separation (more training epochs / bottleneck).")
    elif sil < 0.3:
        print("  ⚠️  Weak separation. Clusters exist but overlap significantly.")
    else:
        print("  ✅ Reasonable cluster separation.")

    if ari < 0.3:
        print("  ⚠️  Low ARI — cluster assignments don't align well with object identity.")
    elif ari > 0.7:
        print("  ✅ High ARI — clusters align well with object types.")

# ── Plot 1: t-SNE coloured by identity ───────────────────────────────────────
print("\n→ Running t-SNE (this may take a moment)...")
sample_n  = min(3000, len(feats_all))
idx       = np.random.choice(len(feats_all), sample_n, replace=False)
feats_sub = feats_all[idx]
ident_sub = np.array(all_identities)[idx]
clust_sub = cluster_ids[idx]

tsne      = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
emb       = tsne.fit_transform(feats_sub)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f"Cluster Analysis — {RUN_NAME} (tag={args.tag})", fontsize=14)

# Left: coloured by object identity
unique_idents = sorted(set(ident_sub))
palette_ident = plt.cm.tab20(np.linspace(0, 1, len(unique_idents)))
ident_color   = {ident: palette_ident[i] for i, ident in enumerate(unique_idents)}

for ident in unique_idents:
    mask = (ident_sub == ident)
    axes[0].scatter(emb[mask, 0], emb[mask, 1],
                    c=[ident_color[ident]], label=ident, s=8, alpha=0.6)
axes[0].set_title("t-SNE coloured by Object Identity")
axes[0].legend(fontsize=7, markerscale=2, loc='best')
axes[0].set_xticks([]); axes[0].set_yticks([])

# Right: coloured by cluster assignment
active_clusters = sorted(set(clust_sub))
palette_clust   = plt.cm.tab20(np.linspace(0, 1, len(active_clusters)))
clust_color     = {cid: palette_clust[i] for i, cid in enumerate(active_clusters)}

for cid in active_clusters:
    mask = (clust_sub == cid)
    code = format(cid, f'0{obj_sym_size}b')
    axes[1].scatter(emb[mask, 0], emb[mask, 1],
                    c=[clust_color[cid]], label=f"[{code}]", s=8, alpha=0.6)
axes[1].set_title("t-SNE coloured by Cluster Assignment")
axes[1].legend(fontsize=7, markerscale=2, loc='best')
axes[1].set_xticks([]); axes[1].set_yticks([])

plt.tight_layout()
tsne_path = os.path.join(log_dir, f"cluster_tsne_{args.tag}.png")
#plt.savefig(tsne_path, dpi=150)
plt.show()
print(f"→ t-SNE plot saved to {tsne_path}")

# ── Plot 2: Heatmap — cluster vs object identity ──────────────────────────────
identity_list = sorted(set(all_identities))
heatmap = np.zeros((K, len(identity_list)), dtype=int)
for cid, ident in zip(cluster_ids, all_identities):
    heatmap[cid, identity_list.index(ident)] += 1

fig, ax = plt.subplots(figsize=(max(8, len(identity_list) * 1.2), max(6, K * 0.5)))
im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(identity_list)))
ax.set_xticklabels(identity_list, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(K))
ax.set_yticklabels([format(i, f'0{obj_sym_size}b') for i in range(K)], fontsize=8)
ax.set_xlabel("Object Identity")
ax.set_ylabel("Cluster Code")
ax.set_title(f"Cluster × Object Identity Heatmap — {RUN_NAME}")
plt.colorbar(im, ax=ax, label="Count")
plt.tight_layout()
heatmap_path = os.path.join(log_dir, f"cluster_heatmap_{args.tag}.png")
plt.savefig(heatmap_path, dpi=150)
plt.show()
print(f"→ Heatmap saved to {heatmap_path}")

# ── Suggestions ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'SUGGESTIONS':^70}")
print("="*70)
n_empty = K - n_active
if n_empty > 0:
    print(f"  ⚠️  {n_empty} empty clusters — K={K} is too large for {len(unique_identities)} object types.")
    print(f"     Try: fit_clusters.py --k {len(unique_identities)} (or a small multiple like {len(unique_identities)*2})")
if n_active > 0:
    sizes  = [len(cluster_to_identities[c]) for c in range(K) if cluster_to_identities[c]]
    imbal  = max(sizes) / (min(sizes) + 1e-9)
    if imbal > 10:
        print(f"  ⚠️  High imbalance (max/min={imbal:.1f}x) — some clusters dominate.")
        print(f"     Try: more k-means iterations (--n_iter 1000) or k-means++ init.")
print("="*70)