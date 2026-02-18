# Stereo Vision and Visual Odometry Project

A complete implementation of Stereo Depth Estimation and Visual Odometry using the KITTI dataset.

## Table of Contents
- [Prerequisites](#prerequisites)
- [How to Run](#how-to-run)
  - [Part A: Stereo Depth](#part-a-stereo-depth)
  - [Part B: Visual Odometry](#part-b-visual-odometry)
- [Project Structure](#project-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [Reports](#reports)

---

## Prerequisites
- Python 3.10+
- `numpy`, `opencv-python`, `matplotlib`, `tqdm`
- Recommended: `uv` for fast package management.

## How to Run

### Part A: Stereo Depth Estimation
Run the script to generate disparity maps, depth maps, and perform the ablation study on matching windows and cost functions (SAD, SSD, NCC).
```bash
uv run main.py
```
**Outputs**:
- `results/result_XX.png`: Qualitative disparity/depth maps.
- Table of Bad-Pixel Rate (BPR) and Mean Absolute Error (MAE) printed in terminal.

### Part B: Stereo Visual Odometry (VO)
Run the VO pipeline on the KITTI Odometry sequences to recover the camera trajectory with full metric scale.
```bash
uv run main_vo.py
```
**Outputs**:
- `results/vo_traj_seq00_ransacTrue_scaleTrue.png`: Trajectory plot vs Ground Truth.
- `results/vo_matches_fXX_ransacTrue.png`: Feature matches highlighting RANSAC inliers.
- Ablation results (RANSAC vs No RANSAC, Metric Scale vs Mono) printed in terminal.

---

## Project Structure
- `src/`: 
  - `stereo_matcher.py`: Core block matching algorithms (SAD, SSD, NCC).
  - `vo.py`: `StereoVO` class for feature-based temporal tracking and triangulation.
  - `utils.py`: Image loading and stereo post-processing (L-R check, hole filling).
  - `utils_vo.py`: Calibration and ground-truth pose utilities for VO.
  - `evaluate.py`: Standard stereo metrics (BPR, MAE).
- `dataset/`: Contains KITTI raw calibration, scene flow training data, and odometry poses.
- `results/`: Directory for algorithm outputs and plots.
- `REPORT.md`: Findings for Stereo Depth estimation.
- `REPORT_VO.md`: Findings for Visual Odometry.

## Evaluation Metrics
- **ATE** (Absolute Trajectory Error): Measuring RMSE between GT and Est camera positions.
- **RPE** (Relative Pose Error): Frame-to-frame drift over a 5-frame window.
- **BPR** (Bad-Pixel Rate): Percentage of disparity pixels exceeding a 3.0px threshold from GT.
