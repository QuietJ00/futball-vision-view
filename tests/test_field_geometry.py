import unittest

import cv2
import numpy as np

from autohl import field_geometry, pitch
from autohl.camera import warp


class FieldGeometryTest(unittest.TestCase):
    meta = {'length': 100, 'width': 60, 'box_depth': 16.4, 'box_half': 18.85,
            'ga_depth': 5.4, 'ga_half': 8.75}

    def make_field(self):
        camera = np.array([45., 58., 24.])
        forward = np.array([60., 0., 0.])-camera
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        rotation = np.stack((right, np.cross(forward, right), forward))
        lens = np.array([[1260., 0., 960.], [0., 1260., 540.], [0., 0., 1.]])
        H = lens @ np.column_stack((rotation[:, :2], -rotation@camera))
        im = np.full((1080, 1920, 3), 45, np.uint8)
        boundary = warp([(0, -30), (100, -30), (100, 30), (0, 30)], H)
        cv2.fillConvexPoly(im, boundary.astype(np.int32), (40, 135, 66))
        for line in pitch.model_lines(self.meta):
            cv2.polylines(im, [np.int32(warp(line, H))], False, (230, 235, 230), 5, cv2.LINE_AA)
        return im, H

    def test_recovers_a_rendered_pitch_without_reference_photos(self):
        im, H = self.make_field()
        moved = np.array([[1., 0., -12.], [0., 1., 6.], [0., 0., 1.]])
        second = cv2.warpPerspective(im, moved, (1920, 1080))
        camera_motion = np.stack((np.eye(3), moved))
        cal = field_geometry.calibrate([im, second], camera_motion, self.meta, 0, log=lambda *_: None)
        self.assertIsNotNone(cal)
        C = pitch.Calib(cal, camera_motion)
        self.assertTrue(C.supports(0, 1))
        # The two goals are visually symmetric, so the solver fixes the sign by the shot.
        flip = np.array([[-1., 0., 100.], [0., 1., 0.], [0., 0., 1.]])
        pred = C.img_from_pitch(0)
        self.assertLess(min(pitch._gap(H, pred, 0), pitch._gap(H@flip, pred, 0)), 6.)
        world = np.array([[30., 2.], [52., 15.]])
        np.testing.assert_allclose(C.to_pitch(1, C.to_img(1, world)), world, atol=1e-4)

    def test_unmarked_turf_has_no_automatic_calibration(self):
        im = np.full((1080, 1920, 3), (40, 135, 66), np.uint8)
        cal = field_geometry.calibrate([im, im], np.stack((np.eye(3), np.eye(3))),
                                       self.meta, 0, log=lambda *_: None)
        self.assertIsNone(cal)


class CentreShiftedTest(unittest.TestCase):
    def test_a_camera_with_its_centre_shifted_is_not_refit(self):
        result = {'valid': np.ones(3, bool), 'centre_shifted': True, 'long_lens': True}
        original = field_geometry.hybrid_calib.refine_long_lens

        def refit(*args, **kwargs):
            raise AssertionError('refit called')

        field_geometry.hybrid_calib.refine_long_lens = refit
        try:
            kept = field_geometry._with_one_camera(result, np.zeros((3, 4, 4, 3)), np.tile(np.eye(3), (3, 1, 1)),
                                                   {'length': 105, 'width': 68}, [], log=lambda *_: None)
        finally:
            field_geometry.hybrid_calib.refine_long_lens = original
        self.assertIs(kept, result)


class CameraRefusalTest(unittest.TestCase):
    def test_a_refusal_by_both_camera_solvers_is_not_overruled(self):
        import os
        F = np.zeros((4, 8, 8, 3), np.uint8)
        cumulative = np.tile(np.eye(3), (4, 1, 1))
        patched = {'solve': field_geometry.field_solver.solve, '_from_long_lens': field_geometry._from_long_lens,
                   '_automatic_anchor': field_geometry._automatic_anchor}

        def single_frame(*args, **kwargs):
            raise AssertionError('single-frame search ran after both camera solvers refused')

        field_geometry.field_solver.solve = lambda *args, **kwargs: None
        field_geometry._from_long_lens = lambda F_, cum, *args: ({}, cum, np.zeros(len(F_), int), None, False)
        field_geometry._automatic_anchor = single_frame
        before = os.environ.get('AUTOHL_CALIBRATOR')
        os.environ['AUTOHL_CALIBRATOR'] = 'camera'
        try:
            result = field_geometry.calibrate(F, cumulative, {'length': 105, 'width': 68}, 0,
                                              inliers=np.full(4, 500), boxes=[np.empty((0, 5))]*4, log=lambda *_: None)
        finally:
            field_geometry.field_solver.solve = patched['solve']
            field_geometry._from_long_lens = patched['_from_long_lens']
            field_geometry._automatic_anchor = patched['_automatic_anchor']
            if before is None:
                os.environ.pop('AUTOHL_CALIBRATOR', None)
            else:
                os.environ['AUTOHL_CALIBRATOR'] = before
        self.assertIsNone(result)
