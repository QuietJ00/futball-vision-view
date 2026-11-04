import unittest

import cv2
import numpy as np

from autohl import camera, speed


class SprintMeasurementTest(unittest.TestCase):
    def test_window_uses_observed_points_and_resists_a_foot_outlier(self):
        xy = np.column_stack((np.arange(31)/6, np.zeros(31)))
        xy[10] = [100., 100.]
        xy[15] += [0., 2.]
        observed = np.ones(31, dtype=bool)
        observed[10] = False
        result = speed.measure_window(xy, observed, 0, 30)
        self.assertAlmostEqual(result['speed_kmh'], 18., places=2)
        self.assertEqual(result['observed_frames'], 30)

    def test_window_rejects_an_unobserved_interval(self):
        observed = np.ones(31, dtype=bool)
        observed[10:16] = False
        with self.assertRaises(ValueError):
            speed.measure_window(np.zeros((31, 2)), observed, 0, 30)

    def test_ground_correction_improves_unseen_features(self):
        rng = np.random.default_rng(8)
        texture = rng.integers(0, 255, (540, 960), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (5, 5), 0)
        frame = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
        transform = np.float32([[1., .004, 1.5], [0., 1., 3.]])
        moved = cv2.warpAffine(frame, transform, (960, 540))
        matches = camera.refine_ground_motion(
            [frame, moved], np.stack((np.eye(3), np.eye(3))), [np.empty((0, 5))]*2,
            0, [0, 1], [(480, 350), (485, 351)],
        )
        self.assertTrue(matches[1]['accepted'])
        self.assertLess(matches[1]['after_px'], .4)
