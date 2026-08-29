import unittest

import numpy as np

from src.config.constants import BEV_SHAPE
from src.geometry.boxes import (
    box_corners_from_parameters,
    box_parameters_from_lidar_corners,
    create_box_corners_camera,
    get_lidar_boxes,
)
from src.geometry.transforms import lidar_xy_to_bev, rectified_camera_to_lidar
from src.targets.bev_targets import create_bev_targets, draw_gaussian_xy, gaussian_2d, gaussian_radius


class GeometryTests(unittest.TestCase):
    def test_identity_transform_and_validation(self):
        points = np.array([[1.0, 2.0, 3.0]])
        transform = np.column_stack((np.eye(3), np.zeros(3)))
        np.testing.assert_allclose(rectified_camera_to_lidar(points, transform, np.eye(3)), points)
        with self.assertRaises(ValueError):
            rectified_camera_to_lidar(np.zeros((3, 2)), transform, np.eye(3))
        with self.assertRaises(ValueError):
            lidar_xy_to_bev(np.zeros((3, 3)))

    def test_box_round_trip_and_invalid_dimensions(self):
        corners = box_corners_from_parameters(10.0, -3.0, 4.0, 2.0, 0.4)
        params = box_parameters_from_lidar_corners(corners)
        self.assertAlmostEqual(params["center_x"], 10.0)
        self.assertAlmostEqual(params["center_y"], -3.0)
        self.assertAlmostEqual(params["length"], 4.0)
        self.assertAlmostEqual(params["width"], 2.0)
        self.assertAlmostEqual(params["yaw"], 0.4)
        with self.assertRaises(ValueError):
            create_box_corners_camera([0, 1, 2], [0, 0, 0], 0)

    def test_roi_filtering(self):
        calib = {
            "tr_velo_to_cam": np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]]),
            "r0_rect": np.eye(3),
        }
        labels = [
            {"type": "Car", "dimensions": [1, 2, 4], "location": [0, 0, 10], "rotation_y": 0},
            {"type": "Car", "dimensions": [1, 2, 4], "location": [0, 0, -10], "rotation_y": 0},
            {"type": "Pedestrian", "dimensions": [1, 1, 1], "location": [0, 0, 10], "rotation_y": 0},
        ]
        self.assertEqual(len(get_lidar_boxes(labels, calib)), 1)


class TargetTests(unittest.TestCase):
    def test_gaussian_edges_and_validation(self):
        for center in ((0, 0), (0, 4), (4, 0), (4, 4)):
            heatmap = np.zeros((5, 5), dtype=np.float32)
            draw_gaussian_xy(heatmap, *center, radius=2)
            self.assertEqual(heatmap[center], 1.0)
        with self.assertRaises(ValueError):
            gaussian_2d((3, 3), sigma=0)
        with self.assertRaises(ValueError):
            draw_gaussian_xy(np.zeros((2, 2)), -1, 0, 1)

    def test_dimension_aware_radius(self):
        self.assertGreater(gaussian_radius(20, 10), gaussian_radius(5, 3))
        small = create_bev_targets([
            {"center_x": 10, "center_y": 0, "length": 1, "width": 0.6, "yaw": 0}
        ])
        large = create_bev_targets([
            {"center_x": 10, "center_y": 0, "length": 8, "width": 3, "yaw": 0}
        ])
        self.assertGreater(
            np.count_nonzero(large["heatmap"]),
            np.count_nonzero(small["heatmap"]),
        )

    def test_empty_boundary_collision_and_malformed_boxes(self):
        empty = create_bev_targets([])
        self.assertEqual(empty["heatmap"].shape, BEV_SHAPE)
        self.assertFalse(empty["regression_mask"].any())

        boxes = [
            {"center_x": 1.01, "center_y": 0.01, "length": 2, "width": 1, "yaw": 0},
            {"center_x": 1.02, "center_y": 0.02, "length": 4, "width": 2, "yaw": 1},
            {"center_x": 70.0, "center_y": 0, "length": 2, "width": 1, "yaw": 0},
        ]
        targets = create_bev_targets(boxes)
        ix, iy = 5, 200
        self.assertEqual(targets["regression_mask"].sum(), 1)
        self.assertEqual(targets["size"][0, ix, iy], 4)
        self.assertAlmostEqual(targets["rotation"][0, ix, iy], np.sin(1))
        with self.assertRaises(ValueError):
            create_bev_targets([{"center_x": 1}])


if __name__ == "__main__":
    unittest.main()
