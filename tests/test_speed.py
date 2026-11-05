import unittest

import cv2
import numpy as np

from autohl import camera, speed


class InstantSpeedTest(unittest.TestCase):
    def run_up(self, seed=3, noise=.1):
        rng = np.random.default_rng(seed)
        t = np.arange(120)/30
        speed_true = np.clip(t*3, 0, 7.)
        x = np.cumsum(speed_true)/30
        cov = np.tile(np.diag([noise**2, noise**2]), (120, 1, 1))
        return np.c_[x, 0*x]+rng.normal(0, noise, (120, 2)), cov, speed_true

    def test_follows_a_changing_speed(self):
        xy, cov, truth = self.run_up()
        found = speed.instant_speed(xy, np.ones(120, bool), cov)['kmh']
        self.assertLess(np.sqrt(np.mean((found[20:100]-truth[20:100]*3.6)**2)), 3.)

    def test_is_more_careful_about_frames_it_did_not_see(self):
        xy, cov, _ = self.run_up()
        seen = np.ones(120, bool)
        seen[60:75] = False
        result = speed.instant_speed(xy, seen, cov)
        self.assertGreater(result['sigma'][67], result['sigma'][40])

    def test_one_jumped_box_is_not_a_sprint(self):
        xy, cov, truth = self.run_up()
        xy[80] += [3., 0.]
        found = speed.instant_speed(xy, np.ones(120, bool), cov)['kmh']
        self.assertLess(abs(found[80]-truth[80]*3.6), 4.)


class PlayerSpeedTest(unittest.TestCase):
    class Line:
        # One metre per pixel along X, known everywhere except frames 40 to 49.
        def supports(self, a, b=None):
            return not 40 <= a < 50

        def to_pitch(self, k, p):
            return np.asarray(p, float)*.05

    def run_at(self, v=8.):
        foot = np.c_[np.arange(120)*v/30/.05, np.zeros(120)]
        return speed.player_speed(self.Line(), foot, np.ones(120, bool), 0, 119)

    def test_steady_run_is_measured(self):
        found = self.run_at()
        self.assertAlmostEqual(np.nanmedian(found['kmh'][60:110]), 28.8, delta=.3)

    def test_no_speed_without_pitch_coordinates(self):
        found = self.run_at()
        self.assertTrue(np.isnan(found['kmh'][45]))
        self.assertTrue(np.isfinite(found['kmh'][20]))

    def test_noise_across_the_run_does_not_add_speed(self):
        rng = np.random.default_rng(2)
        xy = np.c_[np.arange(150)/3., rng.normal(0, .6, 150)]
        cov = np.tile(np.diag([.01, .36]), (150, 1, 1))
        found = speed.instant_speed(xy, np.ones(150, bool), cov)['kmh']
        self.assertLess(abs(np.median(found[30:120])-36.), 1.)


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
