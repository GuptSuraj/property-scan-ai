"""Deterministic room graph initialization and robust pose optimization."""

from collections import defaultdict, deque
import math
import numpy as np
from scipy.optimize import least_squares

from property_scanner.stitching.models import RoomConnectionEvidence, StitchingConfig
from property_scanner.stitching.transforms import components, se2, validate_se2, wrap_angle


def connected_components(room_ids: list[str], edges: list[RoomConnectionEvidence]) -> list[list[str]]:
    adjacency = defaultdict(set)
    for edge in edges:
        adjacency[edge.room_a_id].add(edge.room_b_id)
        adjacency[edge.room_b_id].add(edge.room_a_id)
    remaining, result = set(room_ids), []
    while remaining:
        start = min(remaining)
        seen, queue = {start}, deque([start])
        while queue:
            node = queue.popleft()
            for other in sorted(adjacency[node]):
                if other not in seen:
                    seen.add(other); queue.append(other)
        remaining -= seen
        result.append(sorted(seen))
    return sorted(result, key=lambda part: (-len(part), part))


def choose_root(room_ids: list[str], edges: list[RoomConnectionEvidence], areas: dict[str, float]) -> str:
    scores = {room: [0, 0.0, areas.get(room, 0.0)] for room in room_ids}
    for edge in edges:
        for room in (edge.room_a_id, edge.room_b_id):
            scores[room][0] += 1; scores[room][1] += edge.combined_score
    return min(room_ids, key=lambda room: (-scores[room][0], -scores[room][1], -scores[room][2], room))


def maximum_spanning_tree(nodes: set[str], edges: list[RoomConnectionEvidence]) -> list[RoomConnectionEvidence]:
    parent = {node: node for node in nodes}
    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]; node = parent[node]
        return node
    result = []
    for edge in sorted(edges, key=lambda item: (-item.combined_score, item.connection_id)):
        a, b = find(edge.room_a_id), find(edge.room_b_id)
        if a != b:
            parent[a] = b; result.append(edge)
    return result


def initial_layout(root: str, nodes: set[str], tree: list[RoomConnectionEvidence]) -> dict[str, np.ndarray]:
    adjacency = defaultdict(list)
    for edge in tree:
        relation = validate_se2(edge.relative_transform)
        adjacency[edge.room_a_id].append((edge.room_b_id, relation))
        adjacency[edge.room_b_id].append((edge.room_a_id, np.linalg.inv(relation)))
    poses = {root: np.eye(3)}
    queue = deque([root])
    while queue:
        room = queue.popleft()
        for other, other_to_room in adjacency[room]:
            if other not in poses:
                poses[other] = poses[room] @ other_to_room
                queue.append(other)
    return poses


def edge_residual(edge: RoomConnectionEvidence, poses: dict[str, np.ndarray]) -> np.ndarray:
    expected = validate_se2(edge.relative_transform)
    predicted = np.linalg.inv(poses[edge.room_a_id]) @ poses[edge.room_b_id]
    delta = np.linalg.inv(expected) @ predicted
    x, y, yaw = components(delta)
    return np.array([x, y, yaw])


def reject_cycle_conflicts(edges, tree, poses, threshold):
    tree_ids = {edge.connection_id for edge in tree}
    retained, rejected, norms = list(tree), [], []
    for edge in edges:
        if edge.connection_id in tree_ids:
            continue
        residual = edge_residual(edge, poses)
        norm = float(np.linalg.norm([residual[0], residual[1], residual[2]]))
        norms.append(norm)
        (retained if norm <= threshold else rejected).append(edge)
    return retained, rejected, max(norms, default=0.0)


def optimize_layout(root: str, poses: dict[str, np.ndarray], edges: list[RoomConnectionEvidence], config: StitchingConfig):
    rooms = sorted(room for room in poses if room != root)
    if not rooms or not config.enable_global_optimization:
        residuals = np.concatenate([edge_residual(edge, poses) for edge in edges]) if edges else np.zeros(0)
        return poses, float(np.sqrt(np.mean(residuals**2))) if residuals.size else 0.0
    index = {room: i for i, room in enumerate(rooms)}
    initial = np.array([value for room in rooms for value in components(poses[room])])
    def unpack(values):
        result = {root: np.eye(3)}
        for room, offset in index.items():
            result[room] = se2(*values[offset*3:offset*3+3])
        return result
    def objective(values):
        current = unpack(values)
        rows = []
        for edge in edges:
            weight = math.sqrt(max(edge.combined_score, 1e-3))
            rows.extend(edge_residual(edge, current) * weight)
        return np.asarray(rows)
    fitted = least_squares(objective, initial, loss=config.optimization_loss,
                           max_nfev=config.optimization_max_iterations)
    result = unpack(fitted.x)
    return result, float(np.sqrt(np.mean(objective(fitted.x)**2))) if edges else 0.0

