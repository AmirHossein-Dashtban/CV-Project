import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils import load_image, load_calib, load_disp_gt, post_process, disparity_to_depth
from src.stereo_matcher import StereoMatcher, compute_lr_disparity
from src.evaluate import bad_pixel_rate, mean_absolute_error

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
IMG2_DIR = os.path.join(DATA_DIR, 'data_scene_flow_', 'traning', 'image_2')
IMG3_DIR = os.path.join(DATA_DIR, 'data_scene_flow_', 'traning', 'image_3')
GT_DIR = os.path.join(DATA_DIR, 'data_scene_flow_', 'traning', 'disp_noc_0')
CALIB_DIR = os.path.join(DATA_DIR, 'calib_cam_to_cam')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

def process_frame(frame_id, cost_metric='SAD', window_size=7, max_disp=128):
    """Workflow for a single stereo pair."""
    # Load data
    img_l = load_image(os.path.join(IMG2_DIR, f"{frame_id:06d}_10.png"))
    img_r = load_image(os.path.join(IMG3_DIR, f"{frame_id:06d}_10.png"))
    f, B = load_calib(os.path.join(CALIB_DIR, f"{frame_id:06d}.txt"))
    gt_disp = load_disp_gt(os.path.join(GT_DIR, f"{frame_id:06d}_10.png"))

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

def ablation_study(num_frames=5):
    """Run ablation study on three cost functions and two window sizes over several frames."""
    costs = ['SAD', 'SSD', 'NCC']
    wins = [5, 11]
    
    print(f"\n--- Ablation Study (averaged over {num_frames} frames) ---")
    print(f"{'Metric':<5} | {'Window':<6} | {'Avg Bad-Pixel Rate (%)':<22} | {'Avg MAE':<6}")
    print("-" * 65)
    for c in costs:
        for w in wins:
            total_bpr = 0
            total_mae = 0
            for i in range(num_frames):
                _, _, bpr, mae, _ = process_frame(i, cost_metric=c, window_size=w)
                total_bpr += bpr
                total_mae += mae
            avg_bpr = total_bpr / num_frames
            avg_mae = total_mae / num_frames
            print(f"{c:<5} | {w:<6} | {avg_bpr:<22.2f} | {avg_mae:<6.2f}")

def save_visualizations(n=10):
    """Save n qualitative results."""
    print(f"\n--- Saving Visualizations for {n} frames ---")
    for i in tqdm(range(n)):
        disp, depth, bpr, mae, img_l = process_frame(i, cost_metric='NCC', window_size=11)
        
        # Plotting
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        axes[0].imshow(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Image {i}")
        
        # Disparity visualization (jet)
        disp_viz = (disp * (255.0 / 128.0)).astype(np.uint8)
        disp_viz = cv2.applyColorMap(disp_viz, cv2.COLORMAP_JET)
        axes[1].imshow(cv2.cvtColor(disp_viz, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Disparity (NCC, win=11), BPR={bpr:.2f}%")
        
        # Depth visualization (log scale for better visibility)
        depth_viz = np.log1p(depth)
        axes[2].imshow(depth_viz, cmap='magma')
        axes[2].set_title("Depth Map (Log Scale)")
        
        for ax in axes: ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"result_{i:02d}.png"))
        plt.close()

if __name__ == "__main__":
    # 1. Qualitative results (10 examples)
    save_visualizations(10)
    
    # 2. Ablation study
    ablation_study()
    
    print("\nAll done. Check the 'results/' folder for outputs.")
