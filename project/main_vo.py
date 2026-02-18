import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils_vo import load_calib_vo, load_poses
from src.vo import StereoVO

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

def run_vo(seq="00", num_frames=100, use_ransac=True, use_stereo_scale=True, save_vis=False):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'dataset')
    SEQ_DIR = os.path.join(DATA_DIR, seq)
    POSE_PATH = os.path.join(DATA_DIR, 'poses', f'{seq}.txt')
    
    # Load Calibration and Ground Truth
    K, P0, P1, baseline = load_calib_vo(os.path.join(SEQ_DIR, 'calib.txt'))
    gt_poses_all = load_poses(POSE_PATH)
    
    # Initialize VO state
    vo = StereoVO(K, baseline)
    est_poses = []
    
    processed_count = 0
    print(f"\n--- Running Stereo VO: Seq {seq} ({num_frames} frames) | RANSAC={use_ransac} | Scale={use_stereo_scale} ---")
    
    for i in tqdm(range(num_frames)):
        img_l = cv2.imread(os.path.join(SEQ_DIR, 'image_0', f'{i:06d}.png'), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(os.path.join(SEQ_DIR, 'image_1', f'{i:06d}.png'), cv2.IMREAD_GRAYSCALE)
        
        if img_l is None: break
            
        T_curr, debug_info = vo.process_frame_pair(img_l, img_r, use_ransac=use_ransac, use_stereo_scale=use_stereo_scale)
        est_poses.append(T_curr)
        processed_count += 1
        
        # Save visualization for various frames to show progress
        if save_vis and i in [5, 10, 100, 300, 500, 700] and debug_info:
            prev_img, curr_img, kps_prev, kps_curr, matches, m_mask = debug_info
            vis_img = cv2.drawMatches(prev_img, kps_prev, curr_img, kps_curr, matches, None, 
                                      matchesMask=m_mask.tolist(), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.putText(vis_img, f"RANSAC Inliers: {np.sum(m_mask)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(BASE_DIR, 'results_vo', f'vo_matches_seq{seq}_f{i}_ransac{use_ransac}.png'), vis_img)

    # Truncate GT to match processed frames
    gt_poses = gt_poses_all[:processed_count]
    
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
    results_dir = os.path.join(BASE_DIR, 'results_vo')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'vo_traj_seq{seq}_ransac{use_ransac}_scale{use_stereo_scale}.png'))
    plt.close()
    
    return ate, rpe

if __name__ == "__main__":
    os.makedirs('results_vo', exist_ok=True)
    
    # 1. Sequence 03 (801 frames)
    seq3 = "03"
    n_frames3 = 801
    ate_full3, rpe_full3 = run_vo(seq3, num_frames=n_frames3, use_ransac=True, use_stereo_scale=True, save_vis=True)
    
    # 2. Sequence 01 (1101 frames) as requested
    seq1 = "01"
    n_frames1 = 1101
    ate_full1, rpe_full1 = run_vo(seq1, num_frames=n_frames1, use_ransac=True, use_stereo_scale=True, save_vis=True)
    
    # 3. Ablations on Sequence 03 (for report comparison)
    print("\n--- Running Ablation: No RANSAC ---")
    ate_no_ransac_s3, _ = run_vo("03", num_frames=n_frames3, use_ransac=False, use_stereo_scale=True)
    
    print("\n--- Running Ablation: No Stereo Scale (Monocular) ---")
    ate_no_scale_s3, _ = run_vo("03", num_frames=n_frames3, use_ransac=True, use_stereo_scale=False)
    
    # Print Comparison Table
    print("\n" + "="*60)
    print(f"{'Configuration':<25} | {'ATE (m)':<15} | {'RPE-5 (m)':<15}")
    print("-" * 60)
    print(f"{'Seq 03 (Full)':<25} | {ate_full3:<15.4f} | {rpe_full3:<15.4f}")
    print(f"{'Seq 01 (Full)':<25} | {ate_full1:<15.4f} | {rpe_full1:<15.4f}")
    print(f"{'Ablation: No RANSAC (S3)':<25} | {ate_no_ransac_s3:<15.4f} | {'N/A'}")
    print(f"{'Ablation: No Scale (S3)':<25} | {ate_no_scale_s3:<15.4f} | {'N/A'}")
    print("="*60)
