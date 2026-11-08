import unittest

import numpy as np

from autohl import broadcast_pitch as bp
from autohl import camera, field_solver as fs

META = {'length': 105., 'width': 68.}
TRUE = np.array([.05, .12, 0., np.log(5200.), 50.5, 48., 14.])


def visible_ends(H, index):
    use, a, b = bp._model(META)
    i = use.index(index)
    line = camera.warp(np.linspace(a[i][:2], b[i][:2], 400), H)
    inside = line[(line[:, 0] > 0) & (line[:, 0] < 1920) & (line[:, 1] > 0) & (line[:, 1] < 1080)]
    return np.r_[inside[0], inside[-1]]


class TwoLineCameraTest(unittest.TestCase):
    def test_two_named_lines_recover_a_known_tele_camera(self):
        H = fs._homography(TRUE)
        lines = np.array([visible_ends(H, 2), visible_ends(H, 4)])
        grid = np.stack(np.meshgrid(np.arange(200., 1800, 300), np.arange(200., 1000, 200)), -1).reshape(-1, 2)
        world = camera.warp(grid, np.linalg.inv(H))
        original = bp._positions, bp.FOCALS
        bp._positions, bp.FOCALS = (lambda meta: TRUE[4:7]*[[1, 1, -1]]), np.array([5200.])
        try:
            found, _ = bp.candidates(lines, META)
        finally:
            bp._positions, bp.FOCALS = original
        gaps = [np.median(np.linalg.norm(camera.warp(world, candidate)-grid, axis=1)) for candidate in found]
        self.assertLess(min(gaps), .5)

    def test_two_named_lines_recover_a_known_wide_camera_from_a_low_stand(self):
        # An Alfheim-like camera: 6 m behind the touchline, 9 m up, 870 px lens.
        wide = np.array([.0, .5, 0., np.log(870.), 49.6, 40.3, 9.])
        H = fs._homography(wide)
        lines = np.array([visible_ends(H, 2), visible_ends(H, 4)])
        grid = np.stack(np.meshgrid(np.arange(200., 1800, 300), np.arange(200., 1000, 200)), -1).reshape(-1, 2)
        world = camera.warp(grid, np.linalg.inv(H))
        original = bp._wide_positions, bp.WIDE_FOCALS
        bp._wide_positions, bp.WIDE_FOCALS = (lambda meta: wide[4:7]*[[1, 1, -1]]), np.array([870.])
        try:
            found, focal = bp.candidates(lines, META)
        finally:
            bp._wide_positions, bp.WIDE_FOCALS = original
        wide_found = found[focal < bp.FOCALS[0]]
        self.assertGreater(len(wide_found), 0)
        gaps = [np.median(np.linalg.norm(camera.warp(world, candidate)-grid, axis=1)) for candidate in wide_found]
        self.assertLess(min(gaps), .5)

    def test_the_wide_grid_reaches_lenses_the_tele_grid_does_not(self):
        self.assertLess(bp.WIDE_FOCALS.max(), bp.FOCALS.min()+1e-6)
        self.assertLess(bp.WIDE_FOCALS.min(), 800.)
        self.assertLess(min(bp.WIDE_SETBACKS_M), min(bp.SETBACKS_M))

    def test_pieces_of_one_line_join(self):
        pieces = np.array([[0., 100., 300., 110.], [400., 113.3, 700., 123.3], [100., 500., 100., 900.]])
        lines, weights = bp.long_lines(pieces)
        self.assertEqual(len(lines), 2)
        self.assertGreater(weights[0], 500.)
        self.assertGreater(np.linalg.norm(lines[0][2:]-lines[0][:2]), 690.)


class RefusalTest(unittest.TestCase):
    def test_a_close_up_shows_too_few_people_to_trust(self):
        self.assertFalse(bp._enough_people(np.zeros((2, 2))))
        self.assertTrue(bp._enough_people(np.zeros((bp.MIN_PEOPLE, 2))))


class PinnedTest(unittest.TestCase):
    def test_a_fit_stopped_by_a_bound_is_not_trusted(self):
        free = np.array([0., .3, 0., np.log(4000.), 50., 60., 15.])
        self.assertFalse(bp._pinned(free, META))
        for index, value in ((6, bp.MAX_CAMERA_M), (6, bp.MIN_CAMERA_M), (3, np.log(bp.MAX_FOCAL)),
                             (3, np.log(bp.MIN_FOCAL)), (5, META['width']/2+120)):
            stuck = free.copy()
            stuck[index] = value
            self.assertTrue(bp._pinned(stuck, META), index)


class ShotCutTest(unittest.TestCase):
    def test_a_cut_is_found_and_a_pan_is_not(self):
        green = np.zeros((20, 90, 160, 3), np.uint8)
        green[..., 1] = 140
        green[:, :, 10:20, 2] = 200
        for k in range(20):
            green[k] = np.roll(green[k], k*2, axis=1)
        green[12:, ..., 1] = 30
        green[12:, ..., 0] = 170
        self.assertEqual(camera.shot_cuts(green), [12])


if __name__ == '__main__':
    unittest.main()
