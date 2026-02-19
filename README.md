# Stereo Depth Perception and Visual Odometry

A vision project implemented on the KITTI Vision Benchmark Suite. It is divided into two primary parts: **Stereo Matching** and **Feature-based Visual Odometry (VO)**.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Overview](#project-overview)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Technical Report](#technical-report)

---

## Prerequisites
- Python 3.10+
- `numpy`, `opencv-python`, `matplotlib`, `tqdm`
- Recommended: `uv` for fast package management.

## Project Overview

### Part A: Stereo Depth Estimation
Implements a block-matching algorithm with Winner-Take-All (WTA) strategy using various cost functions:
- **SAD** (Sum of Absolute Differences)
- **SSD** (Sum of Squared Differences)
- **NCC** (Normalized Cross-Correlation)
Includes an ablation study on window sizes (5x5, 11x11, 21x21) and failure case analysis.

### Part B: Stereo Visual Odometry (VO)
Estimates camera pose $T \in SE(3)$ using ORB features as landmarks. The pipeline includes:
- **ORB** extraction and Hamming distance matching.
- **Two-stage RANSAC** (Essential Matrix filtering + PnP motion estimation).
- **Scale Recovery** via stereo triangulation (No Monocular Scale Drift).
Evaluated on KITTI Sequences 01 through 05 using ATE and RPE-5 metrics.

## How to Run

### Run Stereo Depth
```bash
uv run run_stereo.py
```

### Run Visual Odometry
```bash
uv run run_vo.py
```

## Project Structure
- `src/`: 
  - `stereo_matcher.py`: Core block matching algorithms.
  - `vo.py`: `StereoVO` class for trajectory estimation.
  - `utils.py`: Stereo calibration and disparity post-processing.
  - `utils_vo.py`: Metric evaluation scripts (ATE/RPE calculation).
- `dataset/`: KITTI dataset storage.
- `report/`: LaTeX source files for the technical documentation.
- `results/`: Output plots, trajectory maps, and matching visualizations.
- `main.pdf`: Final compiled technical report.

## Technical Report
The comprehensive 14-page technical report (**main.pdf**) contains:
- **Ablation Studies**: Quantitative comparisons (MAE/BPR) for different matching costs and VO configurations.
- **Trajectory Analysis**: Visualized paths for Sequences 01-05 compared to Ground Truth.
- **Visualizations**: Representative disparity maps and ORB feature inliers identified via RANSAC.
- **Failure Analysis**: Identifying top failure cases in depth perception and sources of drift in VO.
