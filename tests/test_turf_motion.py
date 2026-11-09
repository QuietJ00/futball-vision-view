import unittest
from unittest import mock

import cv2
import numpy as np

from autohl import camera, field_geometry as fg


def _turf(width=640, height=360):
    noise = np.random.default_rng(3).integers(0, 255, (height, width), np.uint8)
    texture = cv2.GaussianBlur(noise, (0, 0), 1.5).astype(np.float32)
    texture = (texture-texture.mean())*6+150
    image = np.zeros((height, width, 3), np.uint8)
    image[..., 1] = np.clip(texture, 0, 255)
    image[..., 0] = image[..., 2] = 60
    return image


def _panned(image, shifts):
    height, width = image.shape[:2]
    moves = [np.array([[1, 0, x], [0, 1, 0], [0, 0, 1.]]) for x in shifts]
    frames = [cv2.warpPerspective(image, H, (width, height), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REFLECT) for H in moves]
    return np.array(frames), np.array(moves)


class TurfMotionTest(unittest.TestCase):
    def setUp(self):
        self.frames, self.true = _panned(_turf(), np.arange(8)*4.)
        self.boxes = [np.zeros((0, 5))]*len(self.frames)
        self.points = np.array([[200., 100.], [400., 250.]])

    def test_a_bad_keyframe_link_is_replaced_by_ground_motion(self):
        wrong = self.true.copy()
        wrong[4:] = np.array([[1, 0, 30.], [0, 1, 0], [0, 0, 1]]) @ wrong[4:]
        path, measured = camera.turf_motion(self.frames, self.boxes, wrong)
        self.assertTrue(measured[1:].all())
        gap = camera.warp(self.points, path[-1])-camera.warp(self.points, self.true[-1])
        self.assertLess(np.abs(gap).max(), 1.)

    def test_a_step_the_ground_cannot_measure_keeps_its_link(self):
        blank = np.zeros_like(self.frames)
        blank[..., 1] = 150
        path, measured = camera.turf_motion(blank, self.boxes, self.true)
        self.assertFalse(measured.any())
        np.testing.assert_allclose(path, self.true, atol=1e-9)


class CalibrateTest(unittest.TestCase):
    def setUp(self):
        self.frames = np.zeros((4, 36, 64, 3), np.uint8)
        self.path = np.array([np.eye(3)]*4)
        self.arguments = dict(meta={'length': 100., 'width': 60.}, sky=0, inliers=np.array([0, 90, 90, 90]),
                              boxes=[np.zeros((0, 5))]*4, log=lambda *_: None)

    def calibrate(self):
        return fg.calibrate(self.frames, self.path, **self.arguments)

    def test_a_solver_failure_is_a_refusal(self):
        with (mock.patch.dict('os.environ', {'AUTOHL_CALIBRATOR': 'camera'}),
              mock.patch.object(fg.field_solver, 'solve', side_effect=ValueError('not finite')),
              mock.patch.object(fg, '_automatic_anchor', return_value=None)):
            self.assertIsNone(self.calibrate())

    def test_keys_use_the_scene_path_and_the_frames_between_use_the_ground(self):
        ground = self.path*2
        with (mock.patch.dict('os.environ', {'AUTOHL_CALIBRATOR': 'camera'}),
              mock.patch.object(fg.field_solver, 'solve', return_value={1: np.eye(3)}) as solve,
              mock.patch.object(fg, 'turf_motion', return_value=(ground, np.ones(4, bool))),
              mock.patch.object(fg, '_keys_on_paint', side_effect=lambda *a, **k: a[4]),
              mock.patch.object(fg, '_project_map', return_value=None) as project):
            self.calibrate()
        self.assertIs(solve.call_args.args[1], self.path)
        self.assertIs(project.call_args.args[1], ground)

    def test_only_an_unbroken_ground_path_stands_in_for_paint(self):
        followed = np.ones(400, bool)
        self.assertTrue(fg._followed_from_key(10, 200, followed))
        self.assertFalse(fg._followed_from_key(10, 200, None))
        self.assertFalse(fg._followed_from_key(10, 390, followed))
        followed[50] = False
        self.assertFalse(fg._followed_from_key(10, 200, followed))
        self.assertTrue(fg._followed_from_key(60, 200, followed))


if __name__ == '__main__':
    unittest.main()
