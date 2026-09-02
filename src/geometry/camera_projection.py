import numpy as np


def transformation_to_rectified(
    xyz,
    tr_velo_to_cam,
    r0_rect,
):
    """Transform LiDAR XYZ points into rectified-camera coordinates."""
    xyz = np.asarray(xyz, dtype=np.float64)
    tr_velo_to_cam = np.asarray(
        tr_velo_to_cam,
        dtype=np.float64,
    )
    r0_rect = np.asarray(
        r0_rect,
        dtype=np.float64,
    )

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"xyz must have shape (N, 3), got {xyz.shape}"
        )
    if tr_velo_to_cam.shape != (3, 4):
        raise ValueError(
            "tr_velo_to_cam must have shape (3, 4)"
        )
    if r0_rect.shape != (3, 3):
        raise ValueError(
            "r0_rect must have shape (3, 3)"
        )
    if not np.isfinite(xyz).all():
        raise ValueError(
            "xyz must contain only finite values"
        )
    if (
        not np.isfinite(tr_velo_to_cam).all()
        or not np.isfinite(r0_rect).all()
    ):
        raise ValueError(
            "calibration matrices must be finite"
        )

    xyz_homogeneous = np.column_stack(
        [
            xyz,
            np.ones(len(xyz), dtype=np.float64),
        ]
    )

    camera_xyz = (
        tr_velo_to_cam
        @ xyz_homogeneous.T
    ).T

    rectified_xyz = (
        r0_rect
        @ camera_xyz.T
    ).T

    return rectified_xyz


def project_rectified_to_image(
    rectified_xyz,
    p2,
    image_shape,
):
    """
    Project rectified-camera points into image_2.

    Returns
    -------
    pixels:
        Valid floating-point [u, v] coordinates, shape (M, 2).

    valid_indices:
        Indices into the original LiDAR/rectified point array,
        shape (M,).

    depths:
        Positive rectified-camera z coordinates, shape (M,).
    """
    rectified_xyz = np.asarray(
        rectified_xyz,
        dtype=np.float64,
    )
    p2 = np.asarray(p2, dtype=np.float64)

    if (
        rectified_xyz.ndim != 2
        or rectified_xyz.shape[1] != 3
    ):
        raise ValueError(
            "rectified_xyz must have shape (N, 3)"
        )
    if p2.shape != (3, 4):
        raise ValueError("p2 must have shape (3, 4)")
    if not np.isfinite(p2).all():
        raise ValueError("p2 must contain finite values")
    if len(image_shape) < 2:
        raise ValueError(
            "image_shape must contain height and width"
        )

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])

    if image_height <= 0 or image_width <= 0:
        raise ValueError(
            "image dimensions must be positive"
        )

    original_indices = np.arange(
        len(rectified_xyz),
        dtype=np.int64,
    )

    in_front = (
        np.isfinite(rectified_xyz).all(axis=1)
        & (rectified_xyz[:, 2] > 0)
    )

    visible_points = rectified_xyz[in_front]
    visible_indices = original_indices[in_front]

    visible_homogeneous = np.column_stack(
        [
            visible_points,
            np.ones(
                len(visible_points),
                dtype=np.float64,
            ),
        ]
    )

    image_homogeneous = (
        p2
        @ visible_homogeneous.T
    ).T

    valid_projection = (
        np.isfinite(image_homogeneous).all(axis=1)
        & (image_homogeneous[:, 2] > 0)
    )

    image_homogeneous = image_homogeneous[
        valid_projection
    ]
    visible_points = visible_points[valid_projection]
    visible_indices = visible_indices[valid_projection]

    projected_depth = image_homogeneous[:, 2]
    u = image_homogeneous[:, 0] / projected_depth
    v = image_homogeneous[:, 1] / projected_depth

    inside_image = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < image_width)
        & (v >= 0)
        & (v < image_height)
    )

    pixels = np.column_stack(
        [
            u[inside_image],
            v[inside_image],
        ]
    )
    valid_indices = visible_indices[inside_image]
    depths = visible_points[inside_image, 2]

    return (
        pixels.astype(np.float32),
        valid_indices.astype(np.int64),
        depths.astype(np.float32),
    )


def project_lidar_to_image(
    points,
    calibration,
    image_shape,
):
    """Convenience wrapper for the complete LiDAR-to-image transform."""
    if not isinstance(calibration, dict):
        raise TypeError("calibration must be a dictionary")

    required = {
        "tr_velo_to_cam",
        "r0_rect",
        "p2",
    }
    missing = required.difference(calibration)

    if missing:
        raise KeyError(
            f"missing calibration keys: {sorted(missing)}"
        )

    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "points must have shape (N, 3) or (N, 4)"
        )

    rectified_xyz = transformation_to_rectified(
        points[:, :3],
        calibration["tr_velo_to_cam"],
        calibration["r0_rect"],
    )

    return project_rectified_to_image(
        rectified_xyz,
        calibration["p2"],
        image_shape,
    )
