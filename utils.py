import os
import numpy as np
import torch
from torch_geometric.data import Data

from depth_image_creator import save_unique_depth_images

def print_model_info(model):
    print("Model Architecture:")
    print(model) 

    print("Model Param Count:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    
def update_graph(current_graph, concatenated_image, bounding_box_differences,out_dim):
    if current_graph is None:
        current_graph = Data(
            x=concatenated_image.unsqueeze(0),
            edge_index=torch.empty((2, 0), dtype=torch.long), 
            edge_attr=torch.empty((0, out_dim), dtype=torch.float)
        )
    else:
      
        num_existing_nodes = current_graph.num_nodes
        current_graph.x = torch.cat([current_graph.x, concatenated_image.unsqueeze(0)], dim=0)
        
        new_edges = []
        new_edge_attrs = []
        
        if bounding_box_differences:
            for existing_node, diffs in bounding_box_differences.items():
                
                existing_node_idx = int(existing_node)  # Convert key to index (1-based to 0-based)
                
                # Add edges in both directions: new_node -> existing_node and existing_node -> new_node
                new_edges.append([num_existing_nodes, existing_node_idx]) #new_node -> existing_node
                new_edges.append([existing_node_idx, num_existing_nodes]) #existing_node -> new_node
                
                # Create edge attributes from min_diff and max_diff
                min_diff_0 = diffs['min_diff'][0]
                min_diff_2 = diffs['min_diff'][2]
                max_diff_0 = diffs['max_diff'][0]
                max_diff_2 = diffs['max_diff'][2]
                    
                # z is the 4-dim vector [min0, min2, max0, max2]
                z = [min_diff_0, min_diff_2, max_diff_0, max_diff_2]

                # Concatenate min_diff and max_diff to create the edge attribute (6 values)
                edge_attr = torch.tensor(z, dtype=torch.float)

                # Add the same edge attributes for both directions
                new_edge_attrs.append(edge_attr)
                new_edge_attrs.append(-edge_attr)
        
        # Convert edge list to tensor
        if new_edges:
            new_edges = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
            new_edge_attrs = torch.stack(new_edge_attrs)
            
            # Concatenate the new edges and edge attributes to the existing graph
            current_graph.edge_index = torch.cat(
                [current_graph.edge_index, new_edges.to(current_graph.edge_index.device)],
                dim=1
            )

            current_graph.edge_attr = torch.cat(
                [current_graph.edge_attr, new_edge_attrs.to(current_graph.edge_attr.device)],
                dim=0
            )
    return current_graph


def finalize_graph_with_image(g: Data, out_dim:int, img) -> Data:
    dev = g.x.device
    N   = g.num_nodes

    if g.edge_attr.numel():
        if g.edge_attr.size(1) == out_dim:
            flag = torch.ones(
                (g.edge_attr.size(0), 1),
                dtype=g.edge_attr.dtype, device=dev
            )
            # Ensure flag is on the same device as g.edge_attr
            flag = flag.to(g.edge_attr.device)
            g.edge_attr = torch.cat([g.edge_attr, flag], dim=1)
    else: 
        g.edge_attr = torch.empty((0, out_dim+1), device=dev)

    image_x    = img.unsqueeze(0)
    g.x        = torch.cat([g.x, image_x], dim=0)
    dummy_id   = N 

    if N==1:
        src = torch.arange(N,  device=dev)
        dst = torch.full  ((N,), dummy_id, dtype=torch.long, device=dev)
        new_edges = torch.cat([torch.stack([src, dst]),
                            torch.stack([dst, src])], dim=1)
    else:
        src = torch.arange(1, N, device=dev)
        
        # Ensure the destination tensor matches the new length (N-1)
        num_object_nodes = src.size(0)
        dst = torch.full((num_object_nodes,), dummy_id, dtype=torch.long, device=dev)
        
        # Combine to create bi-directional edges (object -> dummy and dummy -> object)
        new_edges = torch.cat([torch.stack([src, dst]),
                            torch.stack([dst, src])], dim=1)      # (2, 2*(N-1))

    # dummy edge attr
    dummy_attr = torch.zeros((new_edges.size(1), out_dim+1),
                             dtype=g.edge_attr.dtype, device=dev)

    g.edge_index = torch.cat([g.edge_index.to(dev), new_edges], dim=1)
    g.edge_attr  = torch.cat([g.edge_attr.to(dev), dummy_attr], dim=0)

    if g.num_nodes == 1:
        self_edge = torch.tensor([[0], [0]], dtype=torch.long, device=dev)
        self_attr = torch.zeros((1, out_dim + 1), dtype=g.edge_attr.dtype, device=dev)
        g.edge_index = torch.cat([g.edge_index, self_edge], dim=1)
        g.edge_attr  = torch.cat([g.edge_attr,  self_attr],  dim=0)
    return g


def build_bank_optimized(rows, depth_dir, csv_path=None, device=None):
    """
    Loads each unique object image only once and returns a lookup dict.

    Args:
        rows       : list of data rows (dicts with 'object_type', 'object_size_or_path')
        depth_dir  : directory containing upper_*/lower_* .npy depth images
        csv_path   : path to the data CSV, forwarded to save_unique_depth_images (optional)
        device     : torch device (defaults to cuda if available)

    Returns:
        image_cache    : dict {(obj_type, obj_size_raw): tensor [2, 64, 64]}
        match_registry : dict {(obj_type, obj_size_raw): clean_size_str}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if csv_path is not None:
        save_unique_depth_images(csv_path, depth_dir)

    KNOWN_SIZES = [
        0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13,
        0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20
    ]

    match_registry = {}
    image_cache    = {}

    print(f"→ Building unique object image cache from {depth_dir}")

    # --- Step 1: build match registry ---
    for r in rows:
        obj_type     = r.get('object_type')
        obj_size_raw = r.get('object_size_or_path')
        lookup_key   = (obj_type, obj_size_raw)
        if lookup_key not in match_registry:
            try:
                val          = float(obj_size_raw)
                matched_size = min(KNOWN_SIZES, key=lambda x: abs(x - val))
                match_registry[lookup_key] = str(matched_size)
            except ValueError:
                clean = str(obj_size_raw).split('/')[-1].replace('.urdf', '')
                match_registry[lookup_key] = clean

    # --- Step 2: load unique images ---
    unique_keys = set(match_registry.keys())
    print(f"   Found {len(unique_keys)} unique object types to load")

    for obj_type, obj_size_raw in sorted(unique_keys):
        lookup_key = (obj_type, obj_size_raw)
        clean_size = match_registry[lookup_key]
        base_name  = f"{obj_type}_{clean_size}"
        upper_path = os.path.join(depth_dir, f"upper_{base_name}_new.npy")
        lower_path = os.path.join(depth_dir, f"lower_{base_name}_new.npy")

        if os.path.exists(upper_path) and os.path.exists(lower_path):
            u_tensor = torch.from_numpy(np.load(upper_path)).unsqueeze(0).float()
            l_tensor = torch.from_numpy(np.load(lower_path)).unsqueeze(0).float()
            image_cache[lookup_key] = torch.cat((u_tensor, l_tensor), dim=0).to(device)
            print(f"   ✓ Loaded: {base_name}")
        else:
            print(f"   ❌ MISSING: {upper_path}")
            image_cache[lookup_key] = torch.zeros((2, 64, 64), device=device)

    # --- Step 3: verification table ---
    print(f"\n{'='*80}")
    print(f"{'OBJECT MATCHING VERIFICATION REPORT':^80}")
    print(f"{'='*80}")
    print(f"{'Type':<12} | {'Raw Size/Path':<45} | {'Matched Filename'}")
    print(f"{'-'*12}-+-{'-'*45}-+-{'-'*20}")
    for (otype, raw_val), matched in sorted(match_registry.items()):
        display_raw = (str(raw_val)[:42] + '...') if len(str(raw_val)) > 45 else str(raw_val)
        print(f"{str(otype):<12} | {display_raw:<45} | {otype}_{matched}")
    print(f"{'='*80}")
    print(f"Total unique images loaded: {len(image_cache)}\n")

    return image_cache, match_registry