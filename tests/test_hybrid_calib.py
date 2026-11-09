import os
import unittest
from unittest import mock

import numpy as np

from autohl import field_geometry, field_solver as fs, hybrid_calib as hc, pitch, semantic_pitch
from autohl.camera import warp
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


class CalibLensTest(unittest.TestCase):
    def test_coordinates_survive_the_lens_both_ways(self):
        H = np.array([[10., 1., 200.], [.5, -6., 700.], [.0001, .0002, 1.]])
        cal = {'bases': [(0, H, 0, None)], 'meta': sd.META, 'matrix': H[None], 'valid': np.ones(1, bool),
               'segments': np.zeros(1, int), 'k1': .12, 'focal': 1800.}
        c = pitch.Calib(cal, np.eye(3)[None])
        world = np.array([[20., 5.], [60., -12.]])
        np.testing.assert_allclose(c.to_pitch(0, c.to_img(0, world)), world, atol=1e-3)
        self.assertGreater(np.linalg.norm(c.to_img(0, world)-warp(world, H)), .5)


class TurfSlipTest(unittest.TestCase):
    def setUp(self):
        self.motion = np.stack([np.array([[1., 0., 8.*i], [0., 1., 0.], [0., 0., 1.]]) for i in range(6)])
        self.H = np.array([[10., 1., 200.], [.5, -6., 900.], [.0001, .0002, 1.]])
        self.valid = np.ones(6, bool)

    def test_lines_carried_by_the_camera_do_not_slip(self):
        fixed = np.stack([m @ self.H for m in self.motion])
        self.assertLess(field_geometry.turf_slip(fixed, self.valid, self.motion), .01)

    def test_lines_that_jump_are_measured(self):
        moved = np.stack([m @ self.H for m in self.motion])
        moved[3:] = np.array([[1., 0., 5.], [0., 1., 0.], [0., 0., 1.]]) @ moved[3:]
        self.assertAlmostEqual(field_geometry.turf_slip(moved, self.valid, self.motion), 5., delta=.05)


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


class AnchoredLinesTest(unittest.TestCase):
    def test_lines_stay_on_the_ground_between_frames(self):
        clip = sd.make_clip(sd.Scene('t', noise=3.), 42)
        c = hc.calibrate(clip['observations'], clip['cumulative'], sd.META)
        y, x = np.mgrid[500:1000:100, 200:1800:200]
        grid = np.c_[x.ravel(), y.ravel()].astype(float)
        for frame in range(5, 115, 11):
            moved = clip['cumulative'][frame+1] @ np.linalg.inv(clip['cumulative'][frame])
            carried = warp(grid, moved)
            on_ground = warp(grid, c.img_from_pitch(frame+1) @ np.linalg.inv(c.img_from_pitch(frame)))
            self.assertLess(np.median(np.linalg.norm(on_ground-carried, axis=1)), .5)

    def test_thin_keys_do_not_steer_the_lines(self):
        calibration = hc.Calibration([0, 30, 60], {0: np.eye(3), 30: np.eye(3), 60: np.eye(3)}, None, sd.META)
        line = {'segment': 1, 'points': np.zeros((5, 2))}
        kept = hc.anchored(calibration, {0: [line]*2, 30: [line]*5, 60: [line]*6})
        self.assertEqual(list(kept.keys), [30, 60])


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


class BroadcastCameraTest(unittest.TestCase):
    """A camera far back on a long lens that pans along one touchline and zooms in."""
    meta = {'name': 'test', 'length': 101., 'width': 68.}
    position = np.array([50.5, 43., 13.])

    def clip(self, zoom=.06, frames=9):
        focal = np.log(5400.)+np.linspace(0, zoom, frames)
        cameras = [np.r_[.10-.035*k, .22, .01, focal[k], self.position] for k in range(frames)]
        truth = [fs._homography(c) for c in cameras]
        segments = pitch.model_segments(self.meta)
        observations = {}
        for k, H in enumerate(truth):
            lines = []
            for index in (2, 4, 17):
                kind, *shape = segments[index]
                world = np.linspace(shape[0], shape[1], 300) if kind == 'line' else shape[0]
                pixels = warp(world, H)
                pixels = pixels[((pixels >= 0) & (pixels < [1920, 1080])).all(1)]
                if len(pixels) >= 12:
                    lines.append({'segment': index, 'points': pixels[::4], 'confidence': 1.})
            observations[k*12] = lines
        cumulative = np.zeros((12*(frames-1)+1, 3, 3))
        for j in range(len(cumulative)):
            i = min(j//12, frames-2)
            cumulative[j] = pitch.blend(truth[i], truth[i+1], (j-12*i)/12) @ np.linalg.inv(truth[0])
        return observations, cumulative, truth

    def test_long_lens_is_found_without_a_seed(self):
        observations, cumulative, _ = self.clip(zoom=0.)
        c = hc.fit_shared(observations, cumulative, self.meta)
        self.assertIsNotNone(c)
        self.assertLess(np.linalg.norm(c.cameras[0][4:7]-self.position), 2.)
        self.assertEqual(c.k1, 0.)

    def test_zoom_keeps_the_ground_scale(self):
        observations, cumulative, truth = self.clip()
        c = hc.calibrate(observations, cumulative, self.meta)
        self.assertIn('model=zoom', c.notes)
        for k in range(len(truth)-1):
            seen = warp(np.array([[35., -22.], [45., -22.]]), truth[k])
            ground = c.to_pitch(k*12, seen)
            self.assertAlmostEqual(np.linalg.norm(ground[1]-ground[0]), 10., delta=.15)

    def test_a_fit_that_discards_much_of_its_paint_is_refused(self):
        observations, cumulative, _ = self.clip(zoom=0.)
        named = sum(len(lines) for lines in observations.values())
        with mock.patch.object(hc, '_clean', side_effect=lambda seen, *_: (seen, named//2)):
            self.assertIsNone(hc.fit_shared(observations, cumulative, self.meta))
