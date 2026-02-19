import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .utils import load_image, load_calib, load_disp_gt, post_process, disparity_to_depth
from .matcher import StereoMatcher, compute_lr_disparity
from .evaluate import bad_pixel_rate, mean_absolute_error

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STEREO_DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'stereo')
IMG2_DIR = os.path.join(STEREO_DATA_DIR, 'data_scene_flow_', 'original', 'image_2')
IMG3_DIR = os.path.join(STEREO_DATA_DIR, 'data_scene_flow_', 'original', 'image_3')
GT_DIR = os.path.join(STEREO_DATA_DIR, 'data_scene_flow_', 'ground-truth', 'disp_noc_0')
CALIB_DIR = os.path.join(STEREO_DATA_DIR, 'calib_cam_to_cam')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'stereo')

os.makedirs(RESULTS_DIR, exist_ok=True)

def get_all_frames():
    """Get all 200 frame pairs with suffix _10."""
    frame_ids = []
    for i in range(200):
        frame_ids.append((f"{i:06d}", "10"))
    return frame_ids

def process_frame(frame_id_tuple, cost_metric='SAD', window_size=7, max_disp=128):
    """Workflow for a single stereo pair."""
    id_str, suffix = frame_id_tuple
    filename = f"{id_str}_{suffix}.png"
    calib_filename = f"{id_str}.txt"
    
    # Load data
    img_l = load_image(os.path.join(IMG2_DIR, filename))
    img_r = load_image(os.path.join(IMG3_DIR, filename))
    f, B = load_calib(os.path.join(CALIB_DIR, calib_filename))
    gt_disp = load_disp_gt(os.path.join(GT_DIR, filename))

    if img_l is None or img_r is None or gt_disp is None:
        return None

    # Compute Disparity (L-R check)
    disp_l, disp_r = compute_lr_disparity(img_l, img_r, window_size=window_size, max_disparity=max_disp, cost_metric=cost_metric)

    # Post-process
    processed_disp = post_process(disp_l, disp_right=disp_r, do_lr_check=True, do_hole_filling=True)

    # Disparity to Depth
    depth_map = disparity_to_depth(processed_disp, f, B)
    
    # Evaluate
    bpr = bad_pixel_rate(processed_disp, gt_disp)
    mae = mean_absolute_error(processed_disp, gt_disp)
    
    return processed_disp, depth_map, bpr, mae, img_l

def ablation_study(all_frames):
    """Run ablation study on three cost functions and two window sizes over 200 frames."""
    costs = ['SAD', 'SSD', 'NCC']
    wins = [5, 11]
    
    summary = {}
    print(f"\n--- Ablation Study (averaged over {len(all_frames)} frames) ---")
    print(f"{'Metric':<5} | {'Window':<6} | {'Avg Bad-Pixel Rate (%)':<22} | {'Avg MAE':<6}")
    print("-" * 65)
    
    for c in costs:
        summary[c] = {}
        for w in wins:
            total_bpr = 0
            total_mae = 0
            valid_count = 0
            for frame in tqdm(all_frames, desc=f"Ablating {c}-{w}", leave=False):
                result = process_frame(frame, cost_metric=c, window_size=w)
                if result:
                    _, _, bpr, mae, _ = result
                    total_bpr += bpr
                    total_mae += mae
                    valid_count += 1
            
            avg_bpr = total_bpr / valid_count if valid_count > 0 else 0
            avg_mae = total_mae / valid_count if valid_count > 0 else 0
            summary[c][w] = (avg_bpr, avg_mae)
            print(f"{c:<5} | {w:<6} | {avg_bpr:<22.2f} | {avg_mae:<6.2f}")
    
    return summary

def save_visualizations(all_frames):
    """Save 10 frames x 3 methods = 30 visualizations."""
    print(f"\n--- Saving Visualizations for 10 frames and 3 methods ---")
    sub_frames = all_frames[:10]
    methods = ['SAD', 'SSD', 'NCC']
    
    for method in methods:
        for idx, frame in enumerate(sub_frames):
            result = process_frame(frame, cost_metric=method, window_size=11)
            if not result: continue
            
            disp, depth, bpr, mae, img_l = result
            filename_orig = f"{frame[0]}_{frame[1]}.png"
            
            # Plotting
            fig, axes = plt.subplots(3, 1, figsize=(10, 10))
            axes[0].imshow(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"Image {filename_orig} (Method: {method})")
            
            # Disparity visualization (jet)
            disp_viz = (disp * (255.0 / 128.0)).astype(np.uint8)
            disp_viz = cv2.applyColorMap(disp_viz, cv2.COLORMAP_JET)
            axes[1].imshow(cv2.cvtColor(disp_viz, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"Disparity (win=11), BPR={bpr:.2f}%")
            
            # Depth visualization (log scale for better visibility)
            depth_viz = np.log1p(depth)
            axes[2].imshow(depth_viz, cmap='magma')
            axes[2].set_title("Depth Map (Log Scale)")
            
            for ax in axes: ax.axis('off')
            
            plt.tight_layout()
            out_filename = f"vis_{method}_{filename_orig}"
            plt.savefig(os.path.join(RESULTS_DIR, out_filename))
            plt.close()

def find_failures(all_frames):
    """Identify top 10 failures and print in terminal."""
    print(f"\n--- Identifying Top 10 failures ---")
    results = []
    # Use NCC-11 as the benchmark for failures
    for frame in tqdm(all_frames):
        res = process_frame(frame, cost_metric='NCC', window_size=11)
        if res:
            _, _, bpr, mae, _ = res
            results.append((frame, bpr, mae))
            
    # Sort by BPR descending
    results.sort(key=lambda x: x[1], reverse=True)
    top_failures = results[:10]
    
    print(f"\n{'Frame Name':<20} | {'BPR (%)':<10} | {'MAE':<10}")
    print("-" * 45)
    for frame, bpr, mae in top_failures:
        fname = f"{frame[0]}_{frame[1]}.png"
        print(f"{fname:<20} | {bpr:<10.2f} | {mae:<10.2f}")
    
    return top_failures

if __name__ == "__main__":
    frames = get_all_frames()
    
    # 1. Visualization
    save_visualizations(frames)
    
    # 2. Ablation Study
    summary = ablation_study(frames)
    
    # 3. Find Failures
    failures = find_failures(frames)
    
    print("\nProcessing complete. Updates for report:")
    print("Ablation Table Data:")
    for c in summary:
        for w in summary[c]:
            bpr, mae = summary[c][w]
            print(f"{c}, {w}: {bpr:.2f}, {mae:.2f}")
            
    print("\nFailure Table Data:")
    for f, bpr, mae in failures:
        print(f"{f[0]}_{f[1]}.png: {bpr:.2f}, {mae:.2f}")

