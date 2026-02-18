import numpy as np

def bad_pixel_rate(disp_pred, disp_gt, threshold=3.0):
    """
    Computes percentage of bad pixels.
    Pixels with disp_gt == 0 are ignored (usually occluded/unknown).
    """
    mask = disp_gt > 0
    if np.sum(mask) == 0:
        return 0.0
    
    error = np.abs(disp_pred[mask] - disp_gt[mask])
    bad_pixels = np.sum(error > threshold)
    return (bad_pixels / np.sum(mask)) * 100.0

def mean_absolute_error(disp_pred, disp_gt):
    """Computes MAE on valid pixels."""
    mask = disp_gt > 0
    if np.sum(mask) == 0:
        return 0.0
    
    error = np.abs(disp_pred[mask] - disp_gt[mask])
    return np.mean(error)
