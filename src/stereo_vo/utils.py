import numpy as np
import os

def load_poses(path):
    """Loads KITTI ground truth poses (T_w_cam)."""
    poses = []
    with open(path, 'r') as f:
        for line in f:
            T = np.fromstring(line, sep=' ').reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses

def load_calib_vo(path):
    """Loads KITTI calibration file and returns focal length, baseline and projection matrices."""
    calib = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                calib[key.strip()] = np.fromstring(value.strip(), sep=' ')
    
    P0 = calib['P0'].reshape(3, 4)
    P1 = calib['P1'].reshape(3, 4)
    
    f = P0[0, 0]
    cx = P0[0, 2]
    cy = P0[1, 2]
    bf = -P1[0, 3] # Baseline * focal = -P1[0,3] (in KITTI, P1_x is -f*b)
    baseline = bf / f
    
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    
    return K, P0, P1, baseline

def get_3d_points(kp_left, disparity, K, baseline):
    """Triangulates 2D points into 3D using focal length and baseline."""
    f = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Avoid Z=0
    mask = disparity > 0
    z = (f * baseline) / (disparity + 1e-6)
    x = (kp_left[:, 0] - cx) * z / f
    y = (kp_left[:, 1] - cy) * z / f
    
    points_3d = np.stack((x, y, z), axis=-1)
    return points_3d, mask
