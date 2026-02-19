import numpy as np
import cv2
import os

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def load_calib(path):
    """
    Loads KITTI calibration file and returns focal length and baseline.
    Uses camera 2 (left color) and index 3 (right color).
    """
    calib = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                calib[key.strip()] = np.fromstring(value.strip(), sep=' ')
    
    P2 = calib['P_rect_02'].reshape(3, 4)
    P3 = calib['P_rect_03'].reshape(3, 4)
    
    f = P2[0, 0]
    # Baseline B = |(P3[0,3] - P2[0,3]) / f|
    baseline = np.abs((P3[0, 3] - P2[0, 3]) / f)
    
    return f, baseline

def load_disp_gt(path):
    """Loads KITTI disparity ground truth (uint16 PNG)."""
    disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return disp.astype(np.float32) / 256.0

def post_process(disp, disp_right=None, do_lr_check=True, do_hole_filling=True, threshold=1.0):
    """
    Post-processing: L-R check, Median filtering, Hole filling.
    """
    h, w = disp.shape
    processed = disp.copy()
    
    # 1. Left-Right Consistency Check (Vectorized)
    if do_lr_check and disp_right is not None:
        y_indices, x_indices = np.indices((h, w))
        x_right = np.round(x_indices - disp).astype(np.int32)
        
        # Valid range for x_right
        valid_range = (x_right >= 0) & (x_right < w)
        
        # Pull values of disp_right at (y, x-disp_left[y,x])
        d_right = np.zeros_like(disp)
        
        # Proper 2D indexing using valid_range
        rows = y_indices[valid_range]
        cols = x_right[valid_range]
        d_right[valid_range] = disp_right[rows, cols]
        
        # Mark inconsistent pixels
        mask = (np.abs(disp - d_right) > threshold) | (~valid_range) | (disp <= 0)
        processed[mask] = 0
    
    # 2. Median Filter (to reduce speckle noise)
    processed = cv2.medianBlur(processed, 3)
    
    # 3. Hole Filling (Simple interpolation: set zero to nearest non-zero to the left)
    if do_hole_filling:
        # Vectorized version of hole filling across rows
        # Loop over rows because it's only h rows
        for y in range(h):
            row = processed[y]
            mask = row == 0
            if mask.all(): continue
            # Find indices of non-zero elements
            indices = np.where(~mask)[0]
            # Fill with leading value
            row[:indices[0]] = row[indices[0]]
            # Fill between non-zero values
            for idx in range(len(indices) - 1):
                row[indices[idx]:indices[idx+1]] = row[indices[idx]]
            # Fill trailing values
            row[indices[-1]:] = row[indices[-1]]
            processed[y] = row
    
    return processed

def disparity_to_depth(disp, f, B):
    """Converts disparity map to depth map."""
    depth = np.zeros_like(disp)
    mask = disp > 0
    depth[mask] = (f * B) / disp[mask]
    return depth
