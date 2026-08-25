"""Dataset access for the KITTI object-detection benchmark."""

from pathlib import Path

import numpy as np

from src.preprocessing.bev import bev_projection


class KittiDataset:
    def __init__(self, root, split="training"):
        self.root = Path(root)
        self.split = split
        if split not in {"training", "testing"}:
            raise ValueError("split must be 'training' or 'testing'")

        self.velodyne_dir = self.root / split / "velodyne"
        self.label_dir = self.root / split / "label_2"
        self.calib_dir = self.root / split / "calib"
        if not self.velodyne_dir.is_dir():
            raise FileNotFoundError(f"LiDAR directory not found: {self.velodyne_dir}")
        if not self.calib_dir.is_dir():
            raise FileNotFoundError(f"calibration directory not found: {self.calib_dir}")

        self.sample_ids = sorted(p.stem for p in self.velodyne_dir.glob("*.bin"))
        self.has_labels = self.label_dir.is_dir()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        return {
            "id": sample_id,
            "points": self.load_velodyne(sample_id),
            "calib": self.load_calibration(sample_id),
            "labels": self.load_labels(sample_id) if self.has_labels else [],
        }

    def load_velodyne(self, sample_id):
        path = self.velodyne_dir / f"{sample_id}.bin"
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4:
            raise ValueError(f"invalid LiDAR file (float count is not divisible by 4): {path}")
        return raw.reshape(-1, 4)

    def bev_projection(self, points):
        """Backward-compatible access to the canonical BEV implementation."""
        return bev_projection(points)

    def load_calibration(self, sample_id):
        path = self.calib_dir / f"{sample_id}.txt"
        values = {}
        with open(path) as file:
            for line in file:
                if not line.strip():
                    continue
                key, raw_values = line.strip().split(":", maxsplit=1)
                values[key] = np.fromstring(raw_values, sep=" ", dtype=np.float32)

        required = {"P2", "R0_rect", "Tr_velo_to_cam"}
        missing = required.difference(values)
        if missing:
            raise ValueError(f"missing calibration keys {sorted(missing)} in {path}")
        return {
            "p2": values["P2"].reshape(3, 4),
            "r0_rect": values["R0_rect"].reshape(3, 3),
            "tr_velo_to_cam": values["Tr_velo_to_cam"].reshape(3, 4),
        }

    def load_labels(self, sample_id):
        if not self.has_labels:
            return []
        path = self.label_dir / f"{sample_id}.txt"
        objects = []
        with open(path) as file:
            for line in file:
                fields = line.split()
                if len(fields) != 15:
                    raise ValueError(f"expected 15 label fields in {path}, got {len(fields)}")
                objects.append({
                    "type": fields[0],
                    "truncated": float(fields[1]),
                    "occluded": int(fields[2]),
                    "alpha": float(fields[3]),
                    "bbox": np.asarray(fields[4:8], dtype=np.float32),
                    "dimensions": np.asarray(fields[8:11], dtype=np.float32),
                    "location": np.asarray(fields[11:14], dtype=np.float32),
                    "rotation_y": float(fields[14]),
                })
        return objects
