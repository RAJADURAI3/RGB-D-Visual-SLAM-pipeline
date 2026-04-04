import os
import numpy as np
import open3d as o3d


# -----------------------------------------------
# Information Matrix (Better weighting)
# -----------------------------------------------
def create_information_matrix(scale_rot=100.0, scale_trans=10.0):
    """
    Create a 6x6 information matrix with different weights for
    rotation and translation.
    """
    info = np.eye(6)
    info[:3, :3] *= scale_rot       # rotation confidence
    info[3:, 3:] *= scale_trans     # translation confidence
    return info


# -----------------------------------------------
# Transform Validation
# -----------------------------------------------
def is_valid_transform(T, max_translation=5.0):
    """
    Reject unrealistic transforms (helps avoid graph corruption).
    """
    if T is None:
        return False

    t_norm = np.linalg.norm(T[:3, 3])
    return t_norm < max_translation


# -----------------------------------------------
# Build Pose Graph (Odometry Edges)
# -----------------------------------------------
def build_pose_graph(global_poses,
                     scale_rot=100.0,
                     scale_trans=10.0):
    """
    Build pose graph with odometry edges (i -> i+1).

    global_poses: list of 4x4 world_T_cam transforms.
    """
    pg = o3d.pipelines.registration.PoseGraph()

    # Open3D stores cam_T_world (inverse of world_T_cam)
    for T_wc in global_poses:
        node_pose = np.linalg.inv(T_wc)
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(node_pose))

    info = create_information_matrix(scale_rot, scale_trans)

    valid_edges = 0

    for i in range(len(global_poses) - 1):
        T_i = global_poses[i]
        T_j = global_poses[i + 1]

        # relative transform i -> j
        T_ij = np.linalg.inv(T_i) @ T_j

        if not is_valid_transform(T_ij):
            continue

        pg.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i, i + 1, T_ij, info, uncertain=False
            )
        )
        valid_edges += 1

    print(f"[PoseGraph] Nodes: {len(pg.nodes)}, Odometry edges: {valid_edges}")

    return pg


# -----------------------------------------------
# Add Loop Closure Edge
# -----------------------------------------------
def add_loop_closure(pg, i, j, T_ij,
                     scale_rot=200.0,
                     scale_trans=20.0):
    """
    Add loop closure edge between non-consecutive frames.

    uncertain=True → optimizer treats it as noisy constraint.
    """
    if not is_valid_transform(T_ij):
        return False

    info = create_information_matrix(scale_rot, scale_trans)

    pg.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(
            i, j, T_ij, info, uncertain=True
        )
    )

    return True


# -----------------------------------------------
# Optimize Pose Graph
# -----------------------------------------------
def optimize_pose_graph(pg,
                        max_correspondence_distance=0.05,
                        edge_prune_threshold=0.25,
                        preference_loop_closure=1.0,
                        reference_node=0):
    """
    Run global optimization using Levenberg-Marquardt.
    """

    print(f"[Optimization] Starting with {len(pg.edges)} edges...")

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance,
        edge_prune_threshold=edge_prune_threshold,
        preference_loop_closure=preference_loop_closure,
        reference_node=reference_node
    )

    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )

    print("[Optimization] Completed.")

    return pg


# -----------------------------------------------
# Extract Optimized Poses
# -----------------------------------------------
def extract_optimized_global_poses(pg):
    """
    Convert optimized node poses (cam_T_world) → world_T_cam.
    """
    global_poses_opt = []

    for node in pg.nodes:
        T_cw = node.pose
        T_wc = np.linalg.inv(T_cw)
        global_poses_opt.append(T_wc)

    return global_poses_opt


# -----------------------------------------------
# Save Trajectory (CSV)
# -----------------------------------------------
def save_positions_csv(global_poses, out_csv):
    """
    Save camera trajectory:
    frame_idx, x, y, z
    """
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_csv, "w") as f:
        f.write("frame_idx,x,y,z\n")

        for idx, T in enumerate(global_poses):
            x, y, z = T[0, 3], T[1, 3], T[2, 3]
            f.write(f"{idx},{x:.6f},{y:.6f},{z:.6f}\n")

    print(f"[Output] Trajectory saved → {out_csv}")
