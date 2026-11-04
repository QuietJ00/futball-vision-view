import unittest

import numpy as np

from autohl.camera import motion_segments
from autohl.pitch import Calib, MAX_ANCHOR_AGE, anchor_coverage


class CalibrationCoverageTest(unittest.TestCase):
    def setUp(self):
        self.meta = {"length": 100, "width": 60}
        self.anchor = np.array([[10., 0., 200.], [0., 5., 700.], [0., 0., 1.]])

    def build(self, inliers, anchors):
        cumulative = np.repeat(np.eye(3)[None], len(inliers), axis=0)
        segments = motion_segments(inliers, len(cumulative))
        bases = [(frame, homography, 100, .5) for frame, homography in anchors]
        valid = anchor_coverage(bases, cumulative, segments, 0, self.meta)
        return Calib(
            {"bases": bases, "meta": self.meta, "mode": "auto", "valid": valid, "segments": segments},
            cumulative,
        )

    def test_camera_cut_starts_an_independent_calibration(self):
        shifted = self.anchor.copy()
        shifted[0, 2] += 400
        calib = self.build(np.array([0, 120, 0, 0, 120]), [(0, self.anchor), (3, shifted)])
        self.assertTrue(calib.supports(0, 1))
        self.assertTrue(calib.supports(3, 4))
        self.assertFalse(calib.supports(1, 3))
        self.assertTrue(np.isnan(calib.to_pitch(2, [(400, 700)])).all())
        np.testing.assert_allclose(calib.to_img(1, [(0, 0)]), [[200, 700]])
        np.testing.assert_allclose(calib.to_img(4, [(0, 0)]), [[600, 700]])

    def test_disagreeing_anchors_do_not_interpolate(self):
        shifted = self.anchor.copy()
        shifted[0, 2] += 100
        calib = self.build(np.array([0, 120, 120]), [(0, self.anchor), (2, shifted)])
        self.assertTrue(calib.supports(0))
        self.assertTrue(calib.supports(2))
        self.assertFalse(calib.supports(1))

    def test_distant_anchor_does_not_influence_a_nearby_one(self):
        last = 2 * MAX_ANCHOR_AGE + 2
        shifted = self.anchor.copy()
        shifted[0, 2] += 100
        calib = self.build(np.full(last + 1, 120), [(0, self.anchor), (last, shifted)])
        np.testing.assert_allclose(calib.to_img(MAX_ANCHOR_AGE, [(0, 0)]), [[200, 700]])
        self.assertFalse(calib.supports(MAX_ANCHOR_AGE + 1))
