import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
import math
import os

def sample_gumbel_diff(*shape, device='cpu'):
    eps = 1e-20
    u1 = torch.rand(shape, device=device)
    u2 = torch.rand(shape, device=device)
    diff = torch.log(torch.log(u2 + eps) / torch.log(u1 + eps) + eps)
    return diff


def gumbel_sigmoid(logits, T=1.0, hard=True, deterministic=False):
    if deterministic:
        y = logits / T
    else:
        g = sample_gumbel_diff(*logits.shape, device=logits.device)
        y = (g + logits) / T
    s = torch.sigmoid(y)
    if hard:
        s_hard = s.round()
        s = (s_hard - s).detach() + s
    return s


class GumbelSigmoidLayer(nn.Module):
    def __init__(self, hard=True, T=1):
        super(GumbelSigmoidLayer, self).__init__()
        self.hard = hard
        self.T = T
        self.deterministic = False

    def forward(self, x):
        if self.deterministic:
            return (x >= 0.0).float()
        return gumbel_sigmoid(x, self.T, self.hard)


class TGModel(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        symbol_size=2,
        discrete: bool = True,
        symbolic: bool = True,
        obj_symbol_size: int = 4,
    ):
        super(TGModel, self).__init__()

        # GAT encoder layers
        self.symbol_size = symbol_size
        self.obj_symbol_size = obj_symbol_size
        self.discrete = discrete
        self.symbolic = symbolic
        edge_feat_dim = 5
        self.encoded_image_size = 64
        self.gat_input_size = self.encoded_image_size

        self.image_encoder = nn.Sequential(
            # Input: [2, 64, 64]
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # -> [16, 32, 32]
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),              # -> [32, 16, 16]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),              # -> [64, 8, 8]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 1. Normalization for encoder features
        self.obj_feat_norm = nn.LayerNorm(self.encoded_image_size)

        # 2. Denser Bottleneck for 4-bit Symbol Extraction
        self.to_object_symbol = nn.Sequential(
            nn.Linear(self.encoded_image_size, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2), nn.ReLU(),
            nn.Linear(hidden_channels // 2, self.obj_symbol_size)
        )

        # 3. Enhanced Decoder for training grounding
        self.object_decoder = nn.Sequential(
            nn.Linear(self.obj_symbol_size, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2), nn.ReLU(),
            nn.Linear(hidden_channels // 2, hidden_channels),
            nn.LayerNorm(hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, self.encoded_image_size)
        )

        # Cluster centroids buffer — empty until fit_clusters() is called.
        # Registered as a buffer so it travels with the model state dict.
        self.register_buffer("cluster_centroids", torch.empty(0))
        # Binary codes matching each centroid row [K, obj_symbol_size]
        self.register_buffer("cluster_codes", torch.empty(0))

        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(pyg_nn.GATConv(
            self.gat_input_size, hidden_channels, heads=1, concat=True,
            edge_dim=edge_feat_dim, residual=True
        ))

        concat_size = hidden_channels + hidden_channels  # 128

        self.to_symbol = nn.Sequential(
            nn.Linear(concat_size, hidden_channels),
            nn.LayerNorm(hidden_channels), nn.ReLU(),

            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2), nn.ReLU(),

            nn.Linear(hidden_channels // 2, hidden_channels // 4),

            nn.Linear(hidden_channels // 4, self.symbol_size)
        )

        self.bottleneck_norm = nn.LayerNorm(concat_size)
        self.gs_layer = GumbelSigmoidLayer()
        self.gs_obj_layer = GumbelSigmoidLayer()

        if symbolic:
            self.decoder = nn.Sequential(
                nn.Linear(self.symbol_size, hidden_channels // 4),
                nn.LayerNorm(hidden_channels // 4), nn.ReLU(),

                nn.Linear(hidden_channels // 4, hidden_channels // 2),
                nn.LayerNorm(hidden_channels // 2), nn.ReLU(),

                nn.Linear(hidden_channels // 2, hidden_channels),
                nn.Linear(hidden_channels, 5)
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_channels + hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.Linear(hidden_channels, 5)
            )

    # -----------------------------------------------------------------------
    # Object symbol extraction — clustering mode only.
    # Call fit_clusters() (or load_ckpt with saved centroids) before use.
    # -----------------------------------------------------------------------

    def get_object_symbols(self, x):
        if self.cluster_centroids.numel() == 0:
            raise RuntimeError(
                "get_object_symbols() called before fit_clusters(). "
                "Run fit_clusters(loader) first, or load a checkpoint that "
                "already has centroids saved (centroids_best.pt)."
            )
        with torch.no_grad():
            img_feats = self.image_encoder(x)          # [N, 64]

        normed           = F.normalize(img_feats, dim=-1)
        normed_centroids = F.normalize(self.cluster_centroids, dim=-1)
        sim              = normed @ normed_centroids.T  # [N, K]
        nearest          = sim.argmax(dim=-1)           # [N]
        bits             = self.cluster_codes[nearest]  # [N, obj_symbol_size]

        return bits, img_feats


    # -----------------------------------------------------------------------
    # Cluster fitting (run once after training, before deployment)
    # -----------------------------------------------------------------------

    def fit_clusters(self, loader, n_iter=300, k=None):
        """
        Runs k-means++ on image encoder outputs over the given DataLoader.
        K defaults to 2^obj_symbol_size but can be overridden with k=N.
        Centroids are sorted by L2 norm for deterministic code assignment.

        After this call, get_object_symbols() will use clustering mode.
        """
        print("→ Collecting image encoder features for clustering...")
        self.eval()
        feats_list = []
        with torch.no_grad():
            for batch in loader:
                batch_x = batch.x.float().to(next(self.parameters()).device)
                feats = self.image_encoder(batch_x)   # [Total_N, 64]
                # Only keep the last node of each graph (newly added object)
                for i in range(batch.num_graphs):
                    node_ids  = (batch.batch == i).nonzero(as_tuple=True)[0]
                    last_node = node_ids[-1]
                    feats_list.append(feats[last_node].cpu().unsqueeze(0))
        feats_all = torch.cat(feats_list, dim=0)      # [num_samples, 64]

        K = k if k is not None else 2 ** self.obj_symbol_size
        print(f"→ Running k-means++ with K={K} clusters, {n_iter} iterations...")

        # --- k-means++ initialisation ---
        centroids = []
        first_idx = torch.randint(len(feats_all), (1,)).item()
        centroids.append(feats_all[first_idx])
        for _ in range(K - 1):
            c_stack = torch.stack(centroids)          # [c, 64]
            dists   = torch.cdist(feats_all, c_stack) # [N, c]
            min_d2  = dists.min(dim=1).values ** 2    # [N]
            probs   = min_d2 / (min_d2.sum() + 1e-12)
            chosen  = torch.multinomial(probs, 1).item()
            centroids.append(feats_all[chosen])
        centroids = torch.stack(centroids)            # [K, 64]

        for it in range(n_iter):
            normed_f = F.normalize(feats_all, dim=-1)
            normed_c = F.normalize(centroids, dim=-1)
            sim      = normed_f @ normed_c.T          # [N, K]
            assignments = sim.argmax(dim=-1)          # [N]

            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(K, dtype=torch.long)
            for ki in range(K):
                mask = assignments == ki
                counts[ki] = mask.sum()
                if counts[ki] > 0:
                    new_centroids[ki] = feats_all[mask].mean(dim=0)
                else:
                    dists = torch.cdist(feats_all, centroids)
                    min_d = dists.min(dim=1).values
                    new_centroids[ki] = feats_all[min_d.argmax()]

            if torch.allclose(centroids, new_centroids, atol=1e-6):
                print(f"   k-means++ converged at iteration {it+1}")
                break
            centroids = new_centroids

        # Sort centroids by L2 norm for deterministic code assignment
        norms      = centroids.norm(dim=-1)
        sorted_idx = norms.argsort()
        centroids  = centroids[sorted_idx]            # [K, 64]
        counts     = counts                           # just for printing

        n_bits = max(1, math.ceil(math.log2(K))) if K > 1 else 1
        codes  = []
        for i in range(K):
            bits = format(i, f'0{n_bits}b')
            codes.append([int(b) for b in bits])
        codes = torch.tensor(codes, dtype=torch.float32)  # [K, n_bits]

        device = next(self.parameters()).device
        self.cluster_centroids = centroids.to(device)
        self.cluster_codes     = codes.to(device)

        print(f"   Code length   : {n_bits} bits  (K={K})")
        print(f"→ fit_clusters done. Clustering mode is now active.")

    def freeze(self, deterministic=True):
        self.gs_layer.hard = True
        self.gs_layer.deterministic = deterministic
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        x, edge_index, edge_attr,
        query_indices,          # tensor [B]
        new_object_indices
    ):
        """
        B  = number of (graph-node, image, action) relations in this mini-batch.
        Unchanged from original — training behaviour is identical.
        """
        # --- GNN encoder on the whole graph (exactly once) ---
        node_emb = self.image_encoder(x)
        node_emb = self.obj_feat_norm(node_emb)
        for layer in self.gat_layers:
            node_emb, _ = layer(node_emb, edge_index, edge_attr,
                                return_attention_weights=True)
            node_emb = torch.relu(node_emb)

        q_feat   = node_emb[query_indices]
        new_feat = node_emb[new_object_indices]

        concat = torch.cat([q_feat, new_feat], dim=-1)
        concat = self.bottleneck_norm(concat)

        if self.symbolic:
            logits = self.to_symbol(concat)
        else:
            logits = concat

        if self.discrete and self.symbolic:
            symbols = self.gs_layer(logits)
            outputs = self.decoder(symbols)
            return outputs, symbols
        elif self.symbolic:
            symbols = logits
            outputs = self.decoder(symbols)
            return outputs, symbols
        else:
            outputs = self.decoder(logits)
            return outputs, None


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def _read_config(log_dir):
    """Reads train.yaml saved by train_simple.py inside logs/<name>/."""
    import yaml
    config_path = os.path.join(log_dir, "train.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_model_from_config(cfg, device):
    """Instantiates TGModel from the nested yaml config dict."""
    return TGModel(
        input_channels=2,
        hidden_channels=cfg["model"]["hidden_dim"],
        symbol_size=cfg["model"]["symbol_size"],
        obj_symbol_size=cfg["model"]["obj_symbol_size"],
        discrete=cfg["model"]["discrete"],
        symbolic=cfg["model"]["symbolic"],
    ).to(device)


def _find_stage1_ckpt(log_dir):
    """
    Returns the path to the stage1 checkpoint in logs/<name>/.
    Prefers best_epoch_*.pt (most recent by modification time),
    falls back to last.pt.
    """
    from glob import glob
    candidates = sorted(
        glob(os.path.join(log_dir, "best_epoch_*.pt")),
        key=os.path.getmtime
    )
    if candidates:
        return candidates[-1]
    last = os.path.join(log_dir, "last.pt")
    if os.path.exists(last):
        return last
    raise FileNotFoundError(
        f"No stage1 checkpoint found in {log_dir}. "
        f"Expected best_epoch_*.pt or last.pt."
    )


def load_ckpt(name, tag="best", device=None):
    """
    Loads a trained TGModel from logs/<name>/.

    Directory layout expected:
        logs/<name>/
            train.yaml          — config snapshot written by train_simple.py
            best_epoch_*.pt     — stage1 checkpoint (latest best is used)
            last.pt             — stage1 fallback if no best_epoch exists
            centroids_<tag>.pt  — cluster buffers (optional, fit_clusters.py)

    Loading logic (in priority order):
        1. centroids_<tag>.pt exists  → clustering mode
        2. neither                    → stage1 weights only

    Returns:
        model   — frozen, eval-mode TGModel on `device`
        cfg     — raw yaml config dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = os.path.join("logs", name)
    cfg = _read_config(log_dir)
    model = _build_model_from_config(cfg, device)

    # --- Stage 1 ---
    stage1_path = _find_stage1_ckpt(log_dir)
    print(f"load_ckpt: loading stage1 from {os.path.basename(stage1_path)}")
    ckpt1  = torch.load(stage1_path, map_location=device, weights_only=False)
    state1 = ckpt1.get("model_state_dict", ckpt1)
    model.load_state_dict(state1, strict=False)

    # --- Clustering mode (optional) ---
    centroids_path = os.path.join(log_dir, f"centroids_{tag}.pt")
    if os.path.exists(centroids_path):
        centroid_data = torch.load(centroids_path, map_location=device, weights_only=False)
        model.cluster_centroids = centroid_data["centroids"].to(device)
        model.cluster_codes     = centroid_data["codes"].to(device)
        print(f"load_ckpt: clustering mode — {model.cluster_centroids.shape[0]} centroids loaded")
    else:
        print("load_ckpt: no centroids found — call fit_clusters() before using get_object_symbols()")

    # --- Freeze and set deterministic eval mode ---
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.gs_layer.hard          = True
    model.gs_layer.deterministic = True
    model.gs_obj_layer.hard          = True
    model.gs_obj_layer.deterministic = True

    return model, cfg


def save_clusters(model, name, tag="best"):
    """
    Saves cluster centroids and codes to logs/<name>/centroids_<tag>.pt
    so load_ckpt will find them and activate clustering mode automatically.

    Call this after model.fit_clusters(loader).
    """
    assert model.cluster_centroids.numel() > 0, \
        "No clusters fitted yet — call model.fit_clusters(loader) first."
    log_dir = os.path.join("logs", name)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"centroids_{tag}.pt")
    torch.save({
        "centroids": model.cluster_centroids.cpu(),
        "codes":     model.cluster_codes.cpu(),
    }, path)
    print(f"Cluster centroids saved to {path}")