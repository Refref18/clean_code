import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

def analyze_and_visualize(csv_file, save_dir):
    csv_files = glob.glob(csv_file)
    if not csv_files:
        print(f"Error: No CSV files found matching {csv_file}")
        return
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    groups = df.groupby(['object_type', 'object_size_or_path'])
    print(f"Found {len(groups)} unique (type, size) groups.\n")
    for (obj_type, obj_size), group_df in groups:
        print(f"--- Analyzing Group: {obj_type} | {obj_size} ---")
        upper_files = group_df['upper_depth_image_path'].tolist()
        lower_files = group_df['lower_depth_image_path'].tolist()
        all_upper, all_lower, pair_diffs = [], [], []
        for u_path, l_path in zip(upper_files, lower_files):
            if os.path.exists(u_path) and os.path.exists(l_path):
                u_img = np.load(u_path)
                l_img = np.load(l_path)
                all_upper.append(u_img)
                all_lower.append(l_img)
                pair_diffs.append(np.abs(u_img - l_img))
                non_sky_values = u_img[u_img < 99.0]
                if non_sky_values.size == 0:
                    print(f"DEBUG [{obj_type}]: No objects detected, only sky (100.0m) found!")
        if not all_upper:
            continue
        all_upper  = np.array(all_upper)
        all_lower  = np.array(all_lower)
        pair_diffs = np.array(pair_diffs)
        mean_pair_diff = np.mean(pair_diffs)
        max_pair_diff  = np.max(pair_diffs)
        print(f"Cross-View Comparison (Upper vs Lower of same object):")
        print(f"  Max Pixel Difference: {max_pair_diff:.6f}")
        print(f"  Mean Pixel Difference: {mean_pair_diff:.6f}")
        if max_pair_diff < 1e-7:
            print("  CRITICAL: Upper and Lower views are MATHEMATICALLY IDENTICAL.")
        else:
            print("  SUCCESS: Views are different.")
        is_same_upper = np.all(all_upper == all_upper[0])
        is_same_lower = np.all(all_lower == all_lower[0])
        print(f"Identical Images across different samples? Upper: {is_same_upper} | Lower: {is_same_lower}")
        u_max, u_min = np.max(all_upper), np.min(all_upper)
        l_max, l_min = np.max(all_lower), np.min(all_lower)
        print(f"Stats - Upper: [{u_min:.2f}, {u_max:.2f}] | Lower: [{l_min:.2f}, {l_max:.2f}]")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Group: {obj_type} ({obj_size})\nCross-View Diff: {mean_pair_diff:.6e}")
        im0 = axes[0].imshow(all_upper[0], cmap='magma')
        axes[0].set_title("Upper View")
        plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(all_lower[0], cmap='magma')
        axes[1].set_title("Lower View")
        plt.colorbar(im1, ax=axes[1])
        im2 = axes[2].imshow(pair_diffs[0], cmap='hot')
        axes[2].set_title("Difference (Abs)")
        plt.colorbar(im2, ax=axes[2])
        plt.tight_layout()
        plt.show()
        print("-" * 60)


def save_unique_depth_images(csv_file, save_dir):
    """
    Identical to original logic. save_dir replaces the hardcoded SAVE_DIR.
    """
    os.makedirs(save_dir, exist_ok=True)

    csv_files = glob.glob(csv_file)
    if not csv_files:
        print(f"Error: No CSV files found to save .npy data.")
        return

    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    groups = df.groupby(['object_type', 'object_size_or_path'])
    print(f"\nSaving unique pair .npy files to: {save_dir}")

    for (obj_type, obj_size), group_df in groups:
        row    = group_df.iloc[0]
        u_path = row['upper_depth_image_path']
        l_path = row['lower_depth_image_path']

        if os.path.exists(u_path) and os.path.exists(l_path):
            u_img      = np.load(u_path)
            l_img      = np.load(l_path)
            clean_size = str(obj_size).split('/')[-1].replace('.urdf', '')
            base_name  = f"{obj_type}_{clean_size}"
            np.save(os.path.join(save_dir, f"upper_{base_name}_new.npy"), u_img)
            np.save(os.path.join(save_dir, f"lower_{base_name}_new.npy"), l_img)
            print(f"  Saved NPY: {base_name}")


if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/train.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    save_unique_depth_images(cfg["data"]["csv_path"], cfg["data"]["depth_image_dir"])
    analyze_and_visualize(cfg["data"]["csv_path"], cfg["data"]["depth_image_dir"])