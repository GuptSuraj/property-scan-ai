"""Validated neighbor/loop ICP and initialized Open3D pose-graph optimization."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from property_scanner.reconstruction.lidar.fusion import Keyframe
from property_scanner.reconstruction.lidar.models import LidarConfig, RegistrationRecord
from property_scanner.reconstruction.lidar.poses import rotation_degrees, rigid_transform


def register_pair(source: Keyframe, target: Keyframe, config: LidarConfig, *, loop: bool = False) -> RegistrationRecord:
    reg = o3d.pipelines.registration
    initial = np.linalg.inv(target.pose) @ source.pose
    before = reg.evaluate_registration(source.cloud, target.cloud, config.icp_max_correspondence_fine, initial)
    coarse = reg.registration_icp(source.cloud, target.cloud, config.icp_max_correspondence_coarse, initial,
        reg.TransformationEstimationPointToPlane(), reg.ICPConvergenceCriteria(max_iteration=config.icp_max_iterations))
    fitted = reg.registration_icp(source.cloud, target.cloud, config.icp_max_correspondence_fine, coarse.transformation,
        reg.TransformationEstimationPointToPlane(), reg.ICPConvergenceCriteria(max_iteration=config.icp_max_iterations))
    matrix = rigid_transform(fitted.transformation)
    reverse = reg.evaluate_registration(target.cloud, source.cloud, config.icp_max_correspondence_fine, np.linalg.inv(matrix))
    delta = np.linalg.inv(initial) @ matrix
    minimum = config.loop_min_fitness if loop else config.neighbor_min_fitness
    maximum = config.loop_max_rmse if loop else config.neighbor_max_rmse
    reasons = []
    if fitted.fitness < minimum:
        reasons.append("low fitness")
    if fitted.fitness == 0 or fitted.inlier_rmse > maximum:
        reasons.append("insufficient correspondences or high RMSE")
    if loop and min(fitted.fitness, reverse.fitness) < config.loop_min_overlap:
        reasons.append("low bidirectional overlap")
    if np.linalg.norm(delta[:3, 3]) > config.max_transform_translation or rotation_degrees(delta[:3, :3]) > config.max_transform_rotation:
        reasons.append("excessive transform deviation")
    information = reg.get_information_matrix_from_point_clouds(source.cloud, target.cloud, config.icp_max_correspondence_fine, matrix)
    return RegistrationRecord(source_frame=source.frame_id, target_frame=target.frame_id, kind="loop" if loop else "neighbor",
        initial_transform=initial.tolist(), optimized_transform=matrix.tolist(), fitness_before=before.fitness,
        rmse_before=before.inlier_rmse if before.fitness else None, fitness=fitted.fitness,
        inlier_rmse=fitted.inlier_rmse if fitted.fitness else None, reverse_fitness=reverse.fitness,
        information_matrix=information.tolist(), accepted=not reasons, rejection_reason="; ".join(reasons) or None)


def loop_candidates(frames: list[Keyframe], config: LidarConfig) -> list[tuple[int, int]]:
    centers = np.array([frame.pose[:3, 3] for frame in frames])
    tree = cKDTree(centers)
    candidates = []
    for i, frame in enumerate(frames):
        nearby = tree.query_ball_point(centers[i], config.loop_search_radius)
        eligible = [j for j in nearby if i-j >= config.loop_min_frame_separation
                    and rotation_degrees(frames[j].pose[:3, :3].T @ frame.pose[:3, :3]) <= config.loop_orientation_tolerance]
        eligible.sort(key=lambda j: (float(np.linalg.norm(centers[i]-centers[j])), j))
        candidates.extend((j, i) for j in eligible[:config.max_loop_candidates_per_frame])
    return candidates


def optimize(frames: list[Keyframe], config: LidarConfig) -> tuple[list[np.ndarray], list[RegistrationRecord], o3d.pipelines.registration.PoseGraph, int]:
    reg = o3d.pipelines.registration
    graph = reg.PoseGraph()
    for frame in frames:
        graph.nodes.append(reg.PoseGraphNode(frame.pose.copy()))
    records = []
    fallback = 0
    for i in range(len(frames)-1):
        record = register_pair(frames[i], frames[i+1], config)
        records.append(record)
        if record.accepted:
            transform, information = np.array(record.optimized_transform), np.array(record.information_matrix)
        else:
            # Preserve connectivity using an explicitly weak DEVICE constraint, not rejected ICP.
            transform = np.array(record.initial_transform)
            information = np.eye(6)*config.device_prior_information
            fallback += 1
        graph.edges.append(reg.PoseGraphEdge(i, i+1, transform, information, uncertain=False))
    candidates = loop_candidates(frames, config)
    for i, j in candidates:
        record = register_pair(frames[i], frames[j], config, loop=True)
        records.append(record)
        if record.accepted:
            graph.edges.append(reg.PoseGraphEdge(i, j, np.array(record.optimized_transform), np.array(record.information_matrix), uncertain=True))
    reg.global_optimization(graph, reg.GlobalOptimizationLevenbergMarquardt(), reg.GlobalOptimizationConvergenceCriteria(),
        reg.GlobalOptimizationOption(max_correspondence_distance=config.icp_max_correspondence_fine,
                                    edge_prune_threshold=config.pose_graph_edge_prune_threshold, reference_node=0))
    poses = [rigid_transform(node.pose).copy() for node in graph.nodes]
    return poses, records, graph, fallback
