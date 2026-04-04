# RGB-D Visual SLAM Pipeline

Real-time camera pose estimation and dense 3D map reconstruction using ORB feature tracking, Essential Matrix estimation, loop closure detection, and pose graph optimization — evaluated on the TUM RGB-D benchmark.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-green)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)](https://opencv.org)
[![Open3D](https://img.shields.io/badge/Open3D-0.17-orange)](http://www.open3d.org)

---

## Results

Evaluated on **TUM RGB-D benchmark (freiburg1_xyz sequence)** using Umeyama trajectory alignment — the standard evaluation protocol for SLAM systems.

| Method | ATE RMSE (m) | RPE Mean (m) |
|--------|-------------|-------------|
| Raw SLAM | 0.1835 | 0.0305 |
| Optimized (+ Pose Graph) | 0.1817 | 0.0306 |
| Improvement | 0.96% | — |

> ATE = Absolute Trajectory Error. RPE = Relative Pose Error.
> Trajectories aligned using Umeyama method before error computation.
> System uses RGB-D camera only — no IMU.

---

## Architecture

```
RGB-D Frames (TUM Dataset)
         │
         ▼
  ORB Feature Extraction
  (2000 features per frame)
         │
         ▼
  KNN Matching + Lowe's Ratio Test (0.75)
         │
         ▼
  Essential Matrix (RANSAC) → Pose Recovery (R, t)
         │
         ├──► Global Pose Accumulation (Odometry)
         │
         ├──► Loop Closure Detection
         │         └── ORB descriptor similarity
         │             across stored keyframes
         │             every 5 frames
         │
         ▼
  Pose Graph Construction
  ├── Odometry edges (consecutive frames)
  └── Loop closure edges (detected revisits)
         │
         ▼
  Global Optimization
  (Levenberg-Marquardt)
         │
         ▼
  Optimized Trajectory + Dense Colorized 3D Map
         │
         ▼
  REST API (FastAPI) — real-time pose estimation
```

---

## Features

- ORB feature extraction with KNN matching and Lowe's ratio test
- Essential Matrix estimation with RANSAC for robust pose recovery
- Loop closure detection via ORB descriptor similarity scoring
- Pose graph optimization using Open3D Levenberg-Marquardt solver
- Transform validation — rejects unrealistic pose estimates
- Separate information matrices for rotation vs translation
- Dense colorized 3D map reconstruction from RGB-D backprojection
- ATE/RPE evaluation with Umeyama alignment against TUM ground truth
- REST API for real-time pose estimation (FastAPI + Uvicorn)
- Clean Python package architecture — modular and extensible
- Fully configurable via `config.yaml`

---

## Project Structure

```
slam_project/
├── app/
│   ├── __init__.py
│   └── Api.py                  # FastAPI REST endpoints
├── slam/
│   ├── __init__.py
│   ├── slam_pipeline.py        # Core SLAM pipeline
│   ├── loopclosure.py          # Loop closure detection
│   └── posegraph.py            # Pose graph build + optimize
├── services/
│   ├── __init__.py
│   └── slam_service.py         # Service layer with FPS tracking
├── evaluation/
│   ├── __init__.py
│   └── evaluate_compare.py     # ATE/RPE evaluation + plots
├── utils/
│   ├── __init__.py
│   └── io.py                   # Dataset loader, config, intrinsics
├── results/                    # Generated outputs
│   ├── trajectory.csv
│   ├── trajectory_optimized.csv
│   ├── trajectory.png
│   ├── ape.png
│   └── rpe.png
├── config.yaml                 # All parameters
├── dockerfile                  # Container deployment
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/RAJADURAI3/RGB-D-Visual-SLAM-pipeline.git
cd RGB-D-Visual-SLAM-pipeline
pip install -r requirements.txt
```

### 2. Download TUM Dataset

Download `freiburg1_xyz` from:
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download

### 3. Configure

Edit `config.yaml`:

```yaml
dataset:
  path: "path/to/rgbd_dataset_freiburg1_xyz"
  max_frames: 300
```

### 4. Run SLAM Pipeline

```bash
python -c "
from utils.io import load_config, load_tum_dataset, get_intrinsics
from slam.slam_pipeline import run_slam

cfg = load_config('config.yaml')
K = get_intrinsics(cfg)
frames = load_tum_dataset(cfg['dataset']['path'], cfg['dataset']['max_frames'])
poses, lcd = run_slam(frames, K, cfg)
lcd.summary()
"
```

### 5. Evaluate

```bash
python evaluation/evaluate_compare.py --config config.yaml
```

Outputs ATE RMSE, RPE, and saves trajectory/error plots to `results/`.

### 6. Run API

```bash
python -m uvicorn app.Api:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger UI.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System status |
| POST | `/estimate_pose` | Submit image → get camera pose |
| POST | `/reset` | Reset session state |

### Example Response — `/estimate_pose`

```json
{
  "frame": 42,
  "loop_closures": 0,
  "fps": 3.21,
  "translation": {
    "x": 0.1234,
    "y": -0.0521,
    "z": 0.8901
  }
}
```

---

## Docker Deployment

```bash
docker build -t slam-api .
docker run -p 8000:8000 slam-api
```

---

## Known Limitations & Future Work

- **Scale ambiguity** — Essential Matrix estimation is scale-ambiguous without IMU. Next iteration will use PnP with depth channel for metric scale recovery.
- **Loop closure** — Current approach uses ORB descriptor similarity. DBoW2 bag-of-words would improve robustness and reduce false positives.
- **Real-time performance** — No threading currently. Target: parallel feature extraction for 30FPS operation.
- **Dummy depth in API** — Current API demo uses placeholder depth. Production version would accept paired RGB-D frames.
- **Dataset** — Evaluated on freiburg1_xyz (straight trajectory). Testing on freiburg1_desk/room would demonstrate loop closure capability.

---

## Tech Stack

- **Python 3.10**
- **OpenCV 4.x** — ORB features, Essential Matrix, RANSAC
- **Open3D** — Pose graph optimization, point cloud I/O
- **NumPy** — Matrix operations, backprojection geometry
- **FastAPI + Uvicorn** — REST API
- **PyYAML** — Configuration management
- **Pandas + Matplotlib** — Trajectory evaluation and plotting

---

## Author

**Rajadurai** — MSc Data Science
Focused on spatial AI, 3D vision, SLAM, and real-time CV systems.

[GitHub](https://github.com/RAJADURAI3) · [LinkedIn](#)