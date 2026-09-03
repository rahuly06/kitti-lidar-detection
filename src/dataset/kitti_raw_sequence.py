"""Loader for one synchronized and rectified KITTI Raw drive."""

from pathlib import Path

import cv2
import numpy as np


def _read_calibration_file(path):
    values = {}
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue
            key, raw_values = line.strip().split(":", maxsplit=1)
            try:
                parsed = np.asarray(
                    [float(token) for token in raw_values.split()],
                    dtype=np.float32,
                )
            except ValueError:
                continue
            if parsed.size:
                values[key] = parsed
    return values


class KittiRawSequence:
    """Expose a KITTI Raw drive using the detector's sample dictionary format."""

    def __init__(self, date_root, drive):
        self.date_root = Path(date_root)
        self.drive = drive
        self.drive_dir = self.date_root / drive
        self.image_dir = self.drive_dir / "image_02" / "data"
        self.velodyne_dir = self.drive_dir / "velodyne_points" / "data"
        for directory in (self.image_dir, self.velodyne_dir):
            if not directory.is_dir():
                raise FileNotFoundError(f"KITTI Raw directory not found: {directory}")

        image_ids = {path.stem for path in self.image_dir.glob("*.png")}
        lidar_ids = {path.stem for path in self.velodyne_dir.glob("*.bin")}
        if not image_ids or not lidar_ids:
            raise ValueError("KITTI Raw drive contains no camera images or LiDAR scans")
        if image_ids != lidar_ids:
            raise ValueError("KITTI Raw image and LiDAR frame IDs do not match")
        self.sample_ids = sorted(image_ids)
        self.calibration = self._load_calibration()
        self.has_labels = False
        self.load_images = True

    def _load_calibration(self):
        camera = _read_calibration_file(self.date_root / "calib_cam_to_cam.txt")
        velodyne = _read_calibration_file(self.date_root / "calib_velo_to_cam.txt")
        required_camera = {"P_rect_02": 12, "R_rect_00": 9}
        required_velodyne = {"R": 9, "T": 3}
        for key, count in required_camera.items():
            if key not in camera or camera[key].size != count:
                raise ValueError(f"invalid or missing {key} in Raw camera calibration")
        for key, count in required_velodyne.items():
            if key not in velodyne or velodyne[key].size != count:
                raise ValueError(f"invalid or missing {key} in Raw Velodyne calibration")
        transform = np.column_stack((velodyne["R"].reshape(3, 3), velodyne["T"]))
        calibration = {
            "p2": camera["P_rect_02"].reshape(3, 4),
            "r0_rect": camera["R_rect_00"].reshape(3, 3),
            "tr_velo_to_cam": transform.astype(np.float32),
        }
        if not all(np.isfinite(value).all() for value in calibration.values()):
            raise ValueError("KITTI Raw calibration contains non-finite values")
        return calibration

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        lidar_path = self.velodyne_dir / f"{sample_id}.bin"
        raw = np.fromfile(lidar_path, dtype=np.float32)
        if raw.size % 4:
            raise ValueError(f"invalid KITTI Raw LiDAR file: {lidar_path}")
        image_path = self.image_dir / f"{sample_id}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode KITTI Raw image: {image_path}")
        return {
            "id": sample_id,
            "points": raw.reshape(-1, 4),
            "image": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "calib": self.calibration,
            "labels": [],
        }
