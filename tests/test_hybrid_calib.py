import os
import unittest
from unittest import mock

import numpy as np

from autohl import field_geometry, hybrid_calib as hc, pitch, semantic_pitch
from experiments.hybrid_pitch import synthetic_data as sd


def held_out_error(calibration, clip, frame=60):
    world, uv = sd.holdout(clip, frame)
    return np.median(np.linalg.norm(calibration.to_pitch(frame, uv)-world, axis=1))


class BlendTest(unittest.TestCase):
    def test_opposite_signs_of_one_homography_do_not_cancel(self):
        H = np.array([[10., 1., 200.], [0., -6., 700.], [.001, .002, 1.]])
        blended = pitch.blend(H, -H, .5)
        np.testing.assert_allclose(blended/blended[2, 2], H, atol=1e-9)


class DistortionTest(unittest.TestCase):
    def test_round_trip_reaches_the_frame_corners(self):
        points = np.array([[0., 0.], [1919., 1079.], [960., 540.], [100., 900.]])
        back = hc.undistort(hc.distort(points, -.15, 1600.), -.15, 1600.)
        np.testing.assert_allclose(back, points, atol=1e-6)


class SyntheticGeometryTest(unittest.TestCase):
    def test_shared_camera_recovers_position_with_named_lines(self):
        clip = sd.make_clip(sd.Scene('t'), 42)
        c = hc.fit_shared(clip['observations'], clip['cumulative'], sd.META)
        found = np.array(list(c.cameras.values()))[0, 4:7]
        np.testing.assert_allclose(found, clip['cameras'][0][4:7], atol=.15)
        self.assertLess(held_out_error(c, clip), .1)

    def test_too_few_lines_give_no_free_homography(self):
        clip = sd.make_clip(sd.Scene('t', lines='centre_circle'), 42)
        self.assertIsNone(hc.fit_homographies(clip['observations'], clip['cumulative'], sd.META))

    def test_tripod_model_finds_the_lens_offset(self):
        clip = sd.make_clip(sd.Scene('t', offset=.3, noise=.5), 42)
        c = hc.fit_tripod(clip['observations'], clip['cumulative'], sd.META)
        self.assertAlmostEqual(float(c.notes.split('=')[1]), .3, delta=.15)

    def test_moving_camera_does_not_keep_the_one_position_model(self):
        clip = sd.make_clip(sd.Scene('t', drift=10., tripod=False), 42)
        c = hc.calibrate(clip['observations'], clip['cumulative'], sd.META)
        self.assertNotIn('model=tripod', c.notes)
        self.assertNotIn('model=shared', c.notes)
        self.assertLess(held_out_error(c, clip), .3)


class SemanticLinesTest(unittest.TestCase):
    def test_every_class_names_a_model_line(self):
        count = len(pitch.model_segments(sd.META))
        self.assertTrue(all(0 <= s < count for s in semantic_pitch.SEGMENT_OF.values()))

    def test_mirror_swaps_only_the_ends(self):
        self.assertEqual(semantic_pitch.mirrored('left_box_front'), 'right_box_front')
        self.assertEqual(semantic_pitch.mirrored('halfway'), 'halfway')

    def test_missing_weights_are_reported(self):
        with mock.patch.dict(os.environ, {'AUTOHL_LINE_WEIGHTS': '/nonexistent'}):
            with self.assertRaises(FileNotFoundError):
                semantic_pitch.SemanticLines()


class FeatureFlagTest(unittest.TestCase):
    def test_new_calibration_stays_off_by_default(self):
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop('AUTOHL_CALIBRATOR', None)
            self.assertFalse(field_geometry.solver_enabled())
