import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import load_calib_vo, load_poses
from .vo import StereoVO

def compute_ate(gt_poses, est_poses):
    """Computes Absolute Trajectory Error (RMSE)."""
    gt_traj = np.array([p[:3, 3] for p in gt_poses])
    est_traj = np.array([p[:3, 3] for p in est_poses])
    
    # Align trajectories (assuming start at identity for both)
    # T_align = gt_0 * est_0^-1
    T_align = gt_poses[0] @ np.linalg.inv(est_poses[0])
    aligned_traj = np.array([(T_align @ p)[:3, 3] for p in est_poses])
    
    error = np.linalg.norm(gt_traj - aligned_traj, axis=1)
    rmse = np.sqrt(np.mean(error**2))
    return rmse

def compute_rpe(gt_poses, est_poses, step=1):
    """Computes Relative Pose Error (RPE) over a fixed step size."""
    errors = []
    for i in range(len(gt_poses) - step):
        # GT relative motion from i to i+step
        rel_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[i+step]
        # Est relative motion from i to i+step
        rel_est = np.linalg.inv(est_poses[i]) @ est_poses[i+step]
        
        # Error matrix: Difference between Estimated and Ground Truth motion
        error_mat = np.linalg.inv(rel_gt) @ rel_est
        t_err = np.linalg.norm(error_mat[:3, 3])
        errors.append(t_err)
    return np.mean(errors) if errors else 0.0

def run_vo(seq="00", num_frames=10000, use_ransac=True, use_stereo_scale=True, save_vis=False, vis_frames=[10, 100]):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'stereo-vo')
    SEQ_DIR = os.path.join(DATA_DIR, seq)
    POSE_PATH = os.path.join(DATA_DIR, 'poses', f'{seq}.txt')
    
    # Check if pose file exists
    if not os.path.exists(POSE_PATH):
        print(f"Warning: Pose file for sequence {seq} not found at {POSE_PATH}")
        return None, None

    # Load Calibration and Ground Truth
    K, P0, P1, baseline = load_calib_vo(os.path.join(SEQ_DIR, 'calib.txt'))
    gt_poses_all = load_poses(POSE_PATH)
    
    # Initialize VO state
    vo = StereoVO(K, baseline)
    est_poses = []
    
    processed_count = 0
    print(f"\n--- Running Stereo VO: Seq {seq} | RANSAC={use_ransac} | Scale={use_stereo_scale} ---")
    
    results_dir = os.path.join(BASE_DIR, 'results', 'stereo-vo')
    os.makedirs(results_dir, exist_ok=True)

    for i in tqdm(range(num_frames)):
        img_l_path = os.path.join(SEQ_DIR, 'image_0', f'{i:06d}.png')
        img_r_path = os.path.join(SEQ_DIR, 'image_1', f'{i:06d}.png')
        
        img_l = cv2.imread(img_l_path, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(img_r_path, cv2.IMREAD_GRAYSCALE)
        
        if img_l is None: break
            
        T_curr, debug_info = vo.process_frame_pair(img_l, img_r, use_ransac=use_ransac, use_stereo_scale=use_stereo_scale)
        est_poses.append(T_curr)
        processed_count += 1
        
        # Save visualization for specific frames
        if save_vis and i in vis_frames and debug_info:
            prev_img, curr_img, kps_prev, kps_curr, matches, m_mask = debug_info
            vis_img = cv2.drawMatches(prev_img, kps_prev, curr_img, kps_curr, matches, None, 
                                      matchesMask=m_mask.tolist(), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.putText(vis_img, f"RANSAC Inliers: {np.sum(m_mask)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(results_dir, f'vo_matches_seq{seq}_f{i}_ransac{use_ransac}.png'), vis_img)

    # Truncate GT to match processed frames
    gt_poses = gt_poses_all[:processed_count]
    
    if len(est_poses) < len(gt_poses):
        gt_poses = gt_poses[:len(est_poses)]
    
    # Evaluation Metrics
    ate = compute_ate(gt_poses, est_poses)
    rpe = compute_rpe(gt_poses, est_poses, step=5) # reporting RPE-5
    
    # Plotting Trajectory
    gt_traj = np.array([p[:3, 3] for p in gt_poses])
    est_traj = np.array([p[:3, 3] for p in est_poses])
    
    plt.figure(figsize=(10, 6))
    plt.plot(gt_traj[:, 0], gt_traj[:, 2], 'g', label='Ground Truth')
    plt.plot(est_traj[:, 0], est_traj[:, 2], 'b--', label='Estimated VO')
    plt.title(f'VO Trajectory: Seq {seq}\nATE: {ate:.2f}m, RPE-5: {rpe:.2f}m')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'vo_traj_seq{seq}_ransac{use_ransac}_scale{use_stereo_scale}.png'))
    plt.close()
    
    return ate, rpe

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RESULTS_VO_DIR = os.path.join(BASE_DIR, 'results', 'stereo-vo')
    os.makedirs(RESULTS_VO_DIR, exist_ok=True)
    
    seqs = ["01", "02", "03", "04", "05"]
    full_results = {}
    
    # 1. Full runs for sequences 01-05
    for seq in seqs:
        # Save viz for frames 10 and 100 for each sequence
        ate, rpe = run_vo(seq, save_vis=True, vis_frames=[10, 100])
        full_results[seq] = (ate, rpe)

    # 2. Ablations on Sequence 02 and 03
    print("\n--- Running Ablations (Seq 02, 03) ---")
    ab_no_ransac_02, _ = run_vo("02", use_ransac=False, use_stereo_scale=True)
    ab_no_scale_02, _ = run_vo("02", use_ransac=True, use_stereo_scale=False)
    
    ab_no_ransac_03, _ = run_vo("03", use_ransac=False, use_stereo_scale=True)
    ab_no_scale_03, _ = run_vo("03", use_ransac=True, use_stereo_scale=False)
    
    # Print Detailed Comparison Table
    print("\n" + "="*80)
    print(f"{'Sequence/Config':<30} | {'ATE (m)':<15} | {'RPE-5 (m)':<15}")
    print("-" * 80)
    for seq in seqs:
        ate, rpe = full_results[seq]
        print(f"{f'Seq {seq} (Full)':<30} | {ate:<15.4f} | {rpe:<15.4f}")
    
    print("-" * 80)
    print(f"{'Seq 02 (No RANSAC)':<30} | {ab_no_ransac_02:<15.4f} | {'N/A'}")
    print(f"{'Seq 02 (No Scale)':<30} | {ab_no_scale_02:<15.4f} | {'N/A'}")
    print(f"{'Seq 03 (No RANSAC)':<30} | {ab_no_ransac_03:<15.4f} | {'N/A'}")
    print(f"{'Seq 03 (No Scale)':<30} | {ab_no_scale_03:<15.4f} | {'N/A'}")
    print("="*80)
