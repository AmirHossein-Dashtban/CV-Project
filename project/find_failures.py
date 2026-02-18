import os
import cv2
import numpy as np
from tqdm import tqdm
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

def find_failures(n=20):
    print(f"Finding failures across {n} frames...")
    results = []
    for i in tqdm(range(n)):
        img_l = load_image(os.path.join(IMG2_DIR, f"{i:06d}_10.png"))
        img_r = load_image(os.path.join(IMG3_DIR, f"{i:06d}_10.png"))
        gt_disp = load_disp_gt(os.path.join(GT_DIR, f"{i:06d}_10.png"))
        
        # Using default settings
        disp_l, disp_r = compute_lr_disparity(img_l, img_r, window_size=7, max_disparity=128, cost_metric='SAD')
        processed_disp = post_process(disp_l, disp_right=disp_r, do_lr_check=True, do_hole_filling=True)
        
        bpr = bad_pixel_rate(processed_disp, gt_disp)
        mae = mean_absolute_error(processed_disp, gt_disp)
        results.append((i, bpr, mae))
    
    # Sort by BPR
    results.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 Worst Frames (Higher BPR):")
    for i, bpr, mae in results[:5]:
        print(f"Image {i:06d}_10.png: BPR = {bpr:.2f}%, MAE = {mae:.2f}")

if __name__ == "__main__":
    find_failures(20)
