"""Evidence-based SE(2) multi-room layout without modifying room dimensions."""

from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from property_scanner.pipeline.context import ScanContext
from property_scanner.schemas.geometry import BoundingDimensions, ConnectionType, PropertyGeometry, Room, RoomConnection
from property_scanner.schemas.measurements import AreaMeasurement, LengthMeasurement
from property_scanner.schemas.primitives import Point2D, Polygon2D
from property_scanner.schemas.result import ResultWarning
from property_scanner.stitching.graph import (choose_root, connected_components, initial_layout,
    maximum_spanning_tree, optimize_layout, reject_cycle_conflicts)
from property_scanner.stitching.models import (RoomConnectionEvidence, RoomToPropertyTransform,
    StitchingConfig, StitchingDiagnostics)
from property_scanner.stitching.transforms import components, transform_room


@dataclass
class StitchingResult:
    property: PropertyGeometry
    initial_transforms: dict[str, RoomToPropertyTransform]
    transforms: dict[str, RoomToPropertyTransform]
    accepted_evidence: list[RoomConnectionEvidence]
    rejected_evidence: list[RoomConnectionEvidence]
    diagnostics: StitchingDiagnostics
    warnings: list[ResultWarning]


def _polygon(room: Room) -> Polygon:
    if room.polygon is None:
        raise ValueError(f"Room {room.room_id} has no polygon")
    polygon = Polygon([(point.x, point.y) for point in room.polygon.points])
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError(f"Room {room.room_id} has invalid polygon geometry")
    return polygon


class MultiRoomStitcher:
    """Place the root's connected component using supplied, validated evidence."""

    def __init__(self, config: StitchingConfig | None = None) -> None:
        self.config = config or StitchingConfig()

    def stitch(self, property_id: str, rooms: list[Room], evidence: list[RoomConnectionEvidence]) -> StitchingResult:
        if not rooms:
            raise ValueError("Stitching requires at least one reconstructed room")
        room_map = {room.room_id: room for room in rooms}
        if len(room_map) != len(rooms):
            raise ValueError("Room IDs must be unique")
        areas = {room.room_id: float(_polygon(room).area) for room in rooms}
        known = set(room_map)
        candidates, rejected = [], []
        for item in evidence[:self.config.max_candidate_room_pairs]:
            if item.room_a_id not in known or item.room_b_id not in known:
                rejected.append(item.model_copy(update={"accepted": False, "rejection_reason": "unknown_room"}))
            elif item.relative_transform is None:
                rejected.append(item.model_copy(update={"accepted": False,
                    "rejection_reason": item.rejection_reason or "relative_transform_unavailable"}))
            elif not item.accepted or item.combined_score < self.config.connection_acceptance_threshold:
                rejected.append(item.model_copy(update={"accepted": False,
                    "rejection_reason": item.rejection_reason or "below_acceptance_threshold"}))
            else:
                candidates.append(item.model_copy(update={"accepted": True, "rejection_reason": None}))

        components_list = connected_components(list(room_map), candidates)
        root = choose_root(list(room_map), candidates, areas)
        positioned_ids = next(component for component in components_list if root in component)
        component_set = set(positioned_ids)
        component_edges = [edge for edge in candidates
                           if edge.room_a_id in component_set and edge.room_b_id in component_set]
        tree = maximum_spanning_tree(component_set, component_edges)
        initial = initial_layout(root, component_set, tree)
        retained, cycle_rejected, cycle_error = reject_cycle_conflicts(
            component_edges, tree, initial, self.config.cycle_error_threshold)
        rejected.extend(edge.model_copy(update={"accepted": False, "rejection_reason": "cycle_inconsistent"})
                        for edge in cycle_rejected)
        poses, residual = optimize_layout(root, initial, retained, self.config)
        placed = [transform_room(room_map[room_id], poses[room_id]) for room_id in sorted(poses)]
        placed_polygons = {room.room_id: _polygon(room) for room in placed}

        overlap_total, maximum_overlap, material_overlaps = 0.0, 0.0, []
        identifiers = sorted(placed_polygons)
        for index, room_a in enumerate(identifiers):
            for room_b in identifiers[index + 1:]:
                area = float(placed_polygons[room_a].intersection(placed_polygons[room_b]).area)
                ratio = area / min(placed_polygons[room_a].area, placed_polygons[room_b].area)
                overlap_total += area
                maximum_overlap = max(maximum_overlap, area)
                if area > self.config.max_overlap_area_m2 and ratio > self.config.max_overlap_ratio:
                    material_overlaps.append((room_a, room_b, area, ratio))

        union = unary_union(list(placed_polygons.values()))
        footprint_parts = [union] if union.geom_type == "Polygon" else list(union.geoms)
        polygon_models = [Polygon2D(points=[Point2D(x=float(x), y=float(y))
            for x, y in list(part.exterior.coords)[:-1]]) for part in footprint_parts]
        footprint = polygon_models[0] if len(polygon_models) == 1 else None
        bounds = union.bounds
        area_sum = sum(areas[room_id] for room_id in poses)
        connections = [RoomConnection(connection_id=edge.connection_id, room_a_id=edge.room_a_id,
            room_b_id=edge.room_b_id, connection_type=ConnectionType.INFERRED,
            confidence=edge.combined_score) for edge in retained]
        source_ids = {room: [] for room in poses}
        for edge in retained:
            source_ids[edge.room_a_id].append(edge.connection_id)
            source_ids[edge.room_b_id].append(edge.connection_id)
        transforms = {}
        for room_id, pose in poses.items():
            x, y, yaw = components(pose)
            qualities = [edge.combined_score for edge in retained
                         if room_id in (edge.room_a_id, edge.room_b_id)]
            transforms[room_id] = RoomToPropertyTransform(room_id=room_id, translation_x=x,
                translation_y=y, rotation_yaw=yaw,
                quality=float(np.mean(qualities)) if qualities else 1.0,
                source_connection_ids=source_ids[room_id])

        warnings = []
        boundary_distances = [float(placed_polygons[edge.room_a_id].boundary.distance(
            placed_polygons[edge.room_b_id].boundary)) for edge in retained]
        if any(distance > self.config.boundary_distance_tolerance for distance in boundary_distances):
            warnings.append(ResultWarning(code="WEAK_ROOM_CONNECTION",
                message="At least one inferred adjacency leaves room boundaries farther apart than configured tolerance."))
        unresolved = sorted(known - set(poses))
        if unresolved:
            warnings.append(ResultWarning(code="DISCONNECTED_ROOM_GRAPH",
                message=f"No reliable property-frame placement was inferred for: {', '.join(unresolved)}."))
        if cycle_rejected:
            warnings.append(ResultWarning(code="AMBIGUOUS_ROOM_CONNECTION",
                message=f"Rejected {len(cycle_rejected)} connection constraint(s) with inconsistent graph cycles."))
        if material_overlaps:
            warnings.extend([
                ResultWarning(code="EXCESSIVE_ROOM_OVERLAP",
                    message="The inferred layout has material room overlap; no whole-property plan should be published."),
                ResultWarning(code="PROPERTY_LAYOUT_CONFLICT",
                    message="Accepted room constraints do not produce a valid non-overlapping layout."),
            ])
        if len(footprint_parts) > 1:
            warnings.append(ResultWarning(code="PROPERTY_FOOTPRINT_DISCONNECTED",
                message="Positioned room polygons form multiple footprint components; gaps were preserved."))
        if cycle_error > self.config.cycle_error_threshold:
            warnings.append(ResultWarning(code="HIGH_CYCLE_CLOSURE_ERROR",
                message=f"Maximum pre-optimization cycle residual was {cycle_error:.3f}."))

        valid = len(poses) == len(rooms) and not material_overlaps
        diagnostics = StitchingDiagnostics(rooms_total=len(rooms), rooms_positioned=len(poses),
            rooms_unpositioned=len(rooms)-len(poses), candidate_edges=len(evidence),
            accepted_edges=len(retained), rejected_edges=len(rejected), root_room_id=root,
            graph_connected=len(components_list) == 1, connected_components=components_list,
            cycle_residual=cycle_error, overlap_area_total=overlap_total,
            maximum_pair_overlap=maximum_overlap, layout_constraint_residual=residual,
            maximum_connected_boundary_distance=max(boundary_distances, default=None),
            room_area_sum_m2=area_sum, polygon_union_area_m2=float(union.area),
            footprint_component_count=len(footprint_parts), valid_layout=valid)
        geometry = PropertyGeometry(property_id=property_id, rooms=placed,
            room_connections=connections, footprint_polygon=footprint,
            footprint_components=polygon_models,
            total_floor_area=AreaMeasurement(value=area_sum, method="sum of rigidly positioned room polygons"),
            bounding_dimensions=BoundingDimensions(
                length=LengthMeasurement(value=float(bounds[2]-bounds[0]), method="property polygon bounds"),
                width=LengthMeasurement(value=float(bounds[3]-bounds[1]), method="property polygon bounds")),
            metadata={"coordinate_scope": "property_shared", "stitching_performed": True,
                "layout_valid": valid, "root_room_id": root, "unpositioned_room_ids": unresolved,
                "sum_minus_union_area_m2": area_sum-float(union.area)})
        initial_models = {room_id: RoomToPropertyTransform(room_id=room_id,
            translation_x=components(pose)[0], translation_y=components(pose)[1],
            rotation_yaw=components(pose)[2]) for room_id, pose in initial.items()}
        return StitchingResult(geometry, initial_models, transforms, retained, rejected, diagnostics, warnings)


class StitchingEngine:
    """Shared stage adapter; direct stitching uses :class:`MultiRoomStitcher`."""

    def run(self, context: ScanContext) -> ScanContext:
        if context.result is None:
            raise NotImplementedError(
                "Generic context stitching is not implemented without reconstructed typed room geometry and connection evidence."
            )
        return context
