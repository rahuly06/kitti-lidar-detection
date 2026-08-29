import math
import unittest

import torch

from src.inference.decode import decode_predictions
from src.losses.detection_loss import DetectionLoss, focal_heatmap_loss
from src.models.bev_detector import BEVDetector


class ModelLossTests(unittest.TestCase):
    def test_model_shapes_and_backward(self):
        model = BEVDetector()
        self.assertAlmostEqual(float(model.heatmap_head[-1].bias.item()), -2.19, places=5)
        outputs = model(torch.zeros((1, 3, 8, 9)))
        self.assertEqual(outputs["heatmap"].shape, (1, 1, 8, 9))
        for key in ("offset", "size", "rotation"):
            self.assertEqual(outputs[key].shape, (1, 2, 8, 9))

        targets = {key: torch.zeros_like(value) for key, value in outputs.items()}
        targets["regression_mask"] = torch.zeros((1, 1, 8, 9))
        losses = DetectionLoss()(outputs, targets)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()

    def test_mask_normalization_and_validation(self):
        prediction = torch.ones((1, 2, 1, 1))
        target = torch.zeros_like(prediction)
        mask = torch.ones((1, 1, 1, 1))
        self.assertEqual(DetectionLoss.masked_l1_loss(prediction, target, mask).item(), 1)
        self.assertTrue(torch.isfinite(focal_heatmap_loss(torch.zeros((1, 1, 2, 2)), torch.zeros((1, 1, 2, 2)))))
        with self.assertRaises(ValueError):
            DetectionLoss.masked_l1_loss(prediction, target, torch.ones((1, 1, 2, 2)))


class DecodeTests(unittest.TestCase):
    def make_outputs(self):
        return {
            "heatmap": torch.full((1, 1, 4, 5), -10.0),
            "offset": torch.zeros((1, 2, 4, 5)),
            "size": torch.ones((1, 2, 4, 5)),
            "rotation": torch.zeros((1, 2, 4, 5)),
        }

    def test_local_maximum_metric_conversion_and_yaw(self):
        outputs = self.make_outputs()
        outputs["heatmap"][0, 0, 1, 2] = 8
        outputs["heatmap"][0, 0, 1, 3] = 7
        outputs["offset"][0, :, 1, 2] = torch.tensor([0.5, 0.25])
        outputs["size"][0, :, 1, 2] = torch.tensor([4.0, 2.0])
        outputs["rotation"][0, :, 1, 2] = torch.tensor([1.0, 0.0])
        detections = decode_predictions(outputs, 0.5, 10)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0]["center_x"], 0.3)
        self.assertAlmostEqual(detections[0]["center_y"], -39.55)
        self.assertAlmostEqual(detections[0]["yaw"], math.pi / 2)

    def test_metric_duplicate_suppression(self):
        outputs = {
            "heatmap": torch.full((1, 1, 4, 8), -10.0),
            "offset": torch.zeros((1, 2, 4, 8)),
            "size": torch.ones((1, 2, 4, 8)),
            "rotation": torch.zeros((1, 2, 4, 8)),
        }
        outputs["heatmap"][0, 0, 1, 0] = 8
        outputs["heatmap"][0, 0, 1, 4] = 7
        detections = decode_predictions(outputs, 0.5, 10, min_center_distance=1.5)
        self.assertEqual(len(detections), 1)
        detections = decode_predictions(outputs, 0.5, 10, min_center_distance=0)
        self.assertEqual(len(detections), 2)

    def test_invalid_sizes_thresholds_and_batch(self):
        outputs = self.make_outputs()
        outputs["heatmap"][0, 0, 1, 1] = 8
        outputs["size"][0, 0, 1, 1] = -1
        self.assertEqual(decode_predictions(outputs), [])
        with self.assertRaises(ValueError):
            decode_predictions(outputs, score_threshold=2)
        outputs["heatmap"] = outputs["heatmap"].repeat(2, 1, 1, 1)
        with self.assertRaises(ValueError):
            decode_predictions(outputs)


if __name__ == "__main__":
    unittest.main()
