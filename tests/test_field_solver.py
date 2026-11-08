import unittest

import cv2
import numpy as np

from autohl import field_solver as solver


META = {'length': 100., 'width': 60.}


def camera(height=6., focal=1600.):
    return np.array([.1, .1, 0., np.log(focal), 22., 42., height])


def pose(p):
    return (solver._rotation(*p[:3])[None], np.array([[p[4], p[5], -p[6]]]), np.array([np.exp(p[3])]))


def seen_people(p, ground):
    # Foot pixels and heights of people at known pitch positions, seen by camera p.
    H = solver._homography(p)
    R, centre, focal = pose(p)
    feet, heights = [], []
    for x, y in ground:
        foot = H @ [x, y, 1.]
        foot = foot[:2]/foot[2]
        view = R[0] @ (np.array([x, y, -solver.PERSON_HEIGHT])-centre[0])
        head_y = focal[0]*view[1]/view[2]+solver.CENTRE[1]
        feet.append(foot)
        heights.append(foot[1]-head_y)
    return np.array(feet), np.array(heights)


class LawnTest(unittest.TestCase):
    @staticmethod
    def _frame(lower):
        hsv = np.zeros((540, 960, 3), np.uint8)
        hsv[:200] = (10, 120, 140)   # a brown stand
        hsv[200:] = lower
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def test_turf_in_deep_shade_is_still_lawn(self):
        lawn = solver._lawn(self._frame((60, 255, 28)))
        self.assertTrue(lawn[300:500, 100:860].mean() > .95)
        self.assertFalse(lawn[:150].any())

    def test_a_dark_fence_is_not_lawn(self):
        lawn = solver._lawn(self._frame((20, 120, 28)))
        self.assertFalse(lawn[300:500].any())


class PeopleTest(unittest.TestCase):
    ground = [(20, -10), (35, 5), (42, 18), (10, 22), (50, -20), (28, 12)]

    def test_true_camera_predicts_people_heights(self):
        p = camera()
        feet, heights = seen_people(p, self.ground)
        error, on_pitch = solver._people_fit(*pose(p), feet, heights, META)
        self.assertAlmostEqual(float(error[0]), 0., places=5)
        self.assertEqual(float(on_pitch[0]), 1.)

    def test_wrong_height_is_exposed_without_any_paint(self):
        p = camera()
        feet, heights = seen_people(p, self.ground)
        wrong = p.copy()
        wrong[6] = 3.
        error, _ = solver._people_fit(*pose(wrong), feet, heights, META)
        self.assertAlmostEqual(float(error[0]), np.log(2.), places=1)

    def test_camera_placing_feet_off_the_pitch_is_exposed(self):
        p = camera()
        feet, heights = seen_people(p, self.ground)
        moved = p.copy()
        moved[5] += 80
        _, on_pitch = solver._people_fit(*pose(moved), feet, heights, META)
        self.assertLess(float(on_pitch[0]), .5)


class FocalFromMotionTest(unittest.TestCase):
    @staticmethod
    def pan(focal, frames=120, per_frame=.004):
        K = np.array([[focal, 0, 960.], [0, focal, 540.], [0, 0, 1.]])
        out = []
        for i in range(frames):
            c, s = np.cos(i*per_frame), np.sin(i*per_frame)
            out.append(K @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ np.linalg.inv(K))
        return np.stack(out)

    def test_pan_gives_the_lens(self):
        found = solver.focal_from_motion(self.pan(1650.), np.arange(120))
        self.assertLess(abs(found/1650.-1), .03)

    def test_static_camera_says_nothing(self):
        self.assertIsNone(solver.focal_from_motion(np.repeat(np.eye(3)[None], 120, 0), np.arange(120)))


class AcceptanceTest(unittest.TestCase):
    good = dict(keys=3, windows=6, gap=8., people=.03, on_pitch=1., height=5.9, lowest=3.7)

    def test_a_stand_camera_passes(self):
        self.assertIsNone(solver._rejection(**self.good))

    def test_each_check_can_refuse_on_its_own(self):
        for change in ({'keys': 1}, {'gap': 272.}, {'people': -.2}, {'on_pitch': .6}, {'height': 2.4}):
            self.assertIsNotNone(solver._rejection(**{**self.good, **change}), change)

    def test_the_minimum_height_follows_distance_and_lens(self):
        stand = solver.minimum_height((22., 43., 6.), 1600., META)
        self.assertAlmostEqual(stand, 3.1, delta=.2)
        self.assertAlmostEqual(solver.minimum_height((52., 47., 6.), 1600., {'length': 105., 'width': 68.}), 3.8, delta=.1)
        self.assertGreater(solver.minimum_height((-8.5, 18.5, 2.4), 1600., META), 5.)
        self.assertEqual(solver.minimum_height((22., 43., 6.), 5400., META), solver.LOWEST_CAMERA_M)
        self.assertGreater(solver.minimum_height((22., 73., 6.), 1600., META), stand)

    def test_a_very_short_clip_needs_one_key(self):
        self.assertIsNone(solver._rejection(**{**self.good, 'keys': 1, 'windows': 1}))


class VoteTest(unittest.TestCase):
    @staticmethod
    def result(p, score=1.):
        return {'p': p, 'score': score, 'people': 0., 'on_pitch': 1.}

    def test_opposite_ends_of_the_pitch_are_one_camera(self):
        p = camera()
        turned = p.copy()
        turned[4], turned[5] = 100-p[4], -p[5]
        turned[0] += np.pi
        groups = solver._vote({0: [self.result(p)], 50: [self.result(turned)]}, META)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]['best']), 2)

    def test_recurring_position_beats_one_good_window(self):
        real, stray = camera(), camera()
        stray[4] = 70.
        results = {0: [self.result(real, .6)], 50: [self.result(real, .6)], 100: [self.result(stray, 1.)]}
        np.testing.assert_allclose(solver._vote(results, META)[0]['position'], real[4:7])

    def test_impossible_cameras_cast_no_vote(self):
        low, short = camera(height=.5), camera()
        results = {0: [self.result(low)], 50: [{**self.result(short), 'people': .6}]}
        self.assertEqual(solver._vote(results, META), [])
