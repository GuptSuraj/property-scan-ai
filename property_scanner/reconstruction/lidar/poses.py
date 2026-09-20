"""One conversion boundary: optical camera-to-canonical-world rigid poses."""
import numpy as np
from property_scanner.core.exceptions import InvalidInputError


def rigid_transform(value: list[list[float]] | np.ndarray) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise InvalidInputError("Pose must be a numeric rectangular 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise InvalidInputError("Pose must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3)):
        raise InvalidInputError("Pose must be rigid with orthonormal rotation and determinant +1")
    return matrix


def normalize_pose(value, convention: str, source_to_canonical: np.ndarray) -> np.ndarray:
    matrix = rigid_transform(value)
    if convention == "world_to_camera":
        matrix = np.linalg.inv(matrix)
    elif convention != "camera_to_world":
        raise InvalidInputError("Unknown pose convention")
    return source_to_canonical @ matrix


def rotation_degrees(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))))
