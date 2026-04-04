import os
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------
# Config Loader
# -----------------------------------------------
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------------
# Ground Truth Loader
# -----------------------------------------------
def load_gt_positions(gt_path):
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    gt_ts, gt_pos = [], []
    with open(gt_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            ts = float(parts[0])
            pos = [float(parts[1]), float(parts[2]), float(parts[3])]
            gt_ts.append(ts)
            gt_pos.append(pos)

    if len(gt_ts) == 0:
        raise ValueError("No valid ground truth data found")

    return np.array(gt_ts), np.array(gt_pos)


# -----------------------------------------------
# Align GT to trajectory length
# -----------------------------------------------
def align_gt_to_len(gt_ts, gt_pos, n):
    if n == 0:
        return np.empty((0, 3))

    est_ts = np.linspace(gt_ts[0], gt_ts[-1], n)
    aligned = []

    for t in est_ts:
        idx = np.argmin(np.abs(gt_ts - t))
        aligned.append(gt_pos[idx])

    return np.array(aligned)


# -----------------------------------------------
# Umeyama Alignment (CRITICAL)
# -----------------------------------------------
def align_trajectories_umeyama(est, gt):
    assert est.shape == gt.shape

    mu_est = est.mean(axis=0)
    mu_gt = gt.mean(axis=0)

    est_centered = est - mu_est
    gt_centered = gt - mu_gt

    H = est_centered.T @ gt_centered / est.shape[0]
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    var_est = np.var(est_centered, axis=0).sum()
    scale = np.trace(np.diag(S)) / var_est

    t = mu_gt - scale * R @ mu_est

    est_aligned = (scale * R @ est.T).T + t
    return est_aligned


# -----------------------------------------------
# Metrics
# -----------------------------------------------
def compute_ate_rpe(est, gt):
    if len(est) == 0:
        return 0, 0, np.array([]), np.array([])

    errors = np.linalg.norm(est - gt, axis=1)
    ate_rmse = np.sqrt(np.mean(errors ** 2))

    rpe = np.linalg.norm(np.diff(est - gt, axis=0), axis=1)
    rpe_mean = np.mean(rpe)

    return ate_rmse, rpe_mean, errors, rpe


# -----------------------------------------------
# Plot: Trajectory
# -----------------------------------------------
def plot_trajectories(est_raw, est_opt, gt_raw, gt_opt, save_path):
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(est_raw[:, 0], est_raw[:, 1], est_raw[:, 2], label="RAW", linewidth=1.2)
    ax1.plot(gt_raw[:, 0], gt_raw[:, 1], gt_raw[:, 2], label="GT", linewidth=1.2)
    ax1.set_title("RAW vs GT")
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(est_opt[:, 0], est_opt[:, 1], est_opt[:, 2], label="OPT", linewidth=1.2)
    ax2.plot(gt_opt[:, 0], gt_opt[:, 1], gt_opt[:, 2], label="GT", linewidth=1.2)
    ax2.set_title("OPT vs GT")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# -----------------------------------------------
# Plot: APE
# -----------------------------------------------
def plot_ape(ape_raw, ape_opt, save_path):
    plt.figure(figsize=(12, 4))
    plt.plot(ape_raw, label="APE RAW")
    plt.plot(ape_opt, label="APE OPT")
    plt.legend()
    plt.title("APE over time")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# -----------------------------------------------
# Plot: RPE
# -----------------------------------------------
def plot_rpe(rpe_raw, rpe_opt, save_path):
    plt.figure(figsize=(12, 4))
    plt.plot(np.hstack(([0], rpe_raw)), label="RPE RAW")
    plt.plot(np.hstack(([0], rpe_opt)), label="RPE OPT")
    plt.legend()
    plt.title("RPE over time")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# -----------------------------------------------
# Main
# -----------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_path = cfg["dataset"]["path"]
    results_dir = cfg["output"]["results_dir"]

    gt_file = os.path.join(dataset_path, "groundtruth.txt")
    traj_raw = os.path.join(results_dir, "trajectory.csv")
    traj_opt = os.path.join(results_dir, "trajectory_optimized.csv")

    for path in [gt_file, traj_raw, traj_opt]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    print("\nLoading data...")
    gt_ts, gt_pos = load_gt_positions(gt_file)

    raw = pd.read_csv(traj_raw)
    opt = pd.read_csv(traj_opt)

    est_raw = raw[["x", "y", "z"]].values
    est_opt = opt[["x", "y", "z"]].values

    gt_raw = align_gt_to_len(gt_ts, gt_pos, len(est_raw))
    gt_opt = align_gt_to_len(gt_ts, gt_pos, len(est_opt))

    # ---- ALIGNMENT ----
    est_raw_aligned = align_trajectories_umeyama(est_raw, gt_raw)
    est_opt_aligned = align_trajectories_umeyama(est_opt, gt_opt)

    # ---- METRICS ----
    ate_raw, rpe_raw_mean, ape_raw, rpe_raw = compute_ate_rpe(est_raw_aligned, gt_raw)
    ate_opt, rpe_opt_mean, ape_opt, rpe_opt = compute_ate_rpe(est_opt_aligned, gt_opt)

    print("\n=== RESULTS ===")
    print(f"RAW  | ATE RMSE: {ate_raw:.4f} m | RPE: {rpe_raw_mean:.4f} m")
    print(f"OPT  | ATE RMSE: {ate_opt:.4f} m | RPE: {rpe_opt_mean:.4f} m")

    improvement = ((ate_raw - ate_opt) / ate_raw) * 100
    print(f"\nATE Improvement: {improvement:.2f}%")

    os.makedirs(results_dir, exist_ok=True)

    pd.DataFrame({"APE_raw": ape_raw}).to_csv(os.path.join(results_dir, "ape_raw.csv"))
    pd.DataFrame({"APE_opt": ape_opt}).to_csv(os.path.join(results_dir, "ape_opt.csv"))

    # ---- PLOTS ----
    plot_trajectories(est_raw_aligned, est_opt_aligned, gt_raw, gt_opt,
                      os.path.join(results_dir, "trajectory.png"))

    plot_ape(ape_raw, ape_opt,
             os.path.join(results_dir, "ape.png"))

    plot_rpe(rpe_raw, rpe_opt,
             os.path.join(results_dir, "rpe.png"))

    print("\nDone. Use these results in your README.")


if __name__ == "__main__":
    main()
