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
                             (3, np.log(bp.MIN_FOCAL))):
            stuck = free.copy()
            stuck[index] = value
            self.assertTrue(bp._pinned(stuck, META), index)

    def test_a_camera_against_the_setback_limit_is_not_called_pinned(self):
        near = np.array([0., .3, 0., np.log(4800.), 50., META['width']/2+3, 12.7])
        self.assertFalse(bp._pinned(near, META))


ALFHEIM = np.array([.0, .5, 0., np.log(870.), 49.6, 40.3, 9.])   # wide camera, low stand


def pitch_segment(index):
    from autohl import pitch
    kind, a, b = pitch.model_segments(META)[index]
    return kind, np.asarray(a, float), np.asarray(b, float)


def model_line(index):
    _, a, b = pitch_segment(index)
    tangent = (b-a)/np.linalg.norm(b-a)
    normal = np.array([-tangent[1], tangent[0]])
    return np.r_[normal, -normal @ a]


def stroke_on(H, index, start=.3, end=.7):
    _, a, b = pitch_segment(index)
    return camera.warp(np.array([a+(b-a)*start, a+(b-a)*end]), H).ravel()


class PolishTest(unittest.TestCase):
    def test_a_principal_point_shift_moves_the_image_and_nothing_else(self):
        world = np.array([[52.5, 0.], [88.5, -20.16], [10., 30.]])
        plain = camera.warp(world, fs._homography(ALFHEIM))
        shifted = camera.warp(world, fs._homography(np.r_[ALFHEIM, 20., -10.]))
        np.testing.assert_allclose(shifted-plain, np.tile([20., -10.], (3, 1)), atol=1e-3)

    def test_a_stroke_past_paint_reach_is_tied_to_the_line_it_runs_along(self):
        # The far penalty box line where a camera with its optical centre 30 px off puts it, seen
        # through a camera that assumes the centre: too far for paint to be tied by distance.
        stroke = stroke_on(fs._homography(np.r_[ALFHEIM, 0., 30.]), 12)
        gap = np.abs(fs._line_error(ALFHEIM, stroke.reshape(2, 2), np.tile(model_line(12), (2, 1))))
        self.assertGreater(gap.min(), 10.)
        points, lines = bp.stroke_ties(ALFHEIM, [stroke], META, 40.)
        self.assertGreater(len(points), 0)
        for line in lines:
            self.assertTrue(np.allclose(line, model_line(12)) or np.allclose(line, -model_line(12)))

    def test_a_stroke_across_every_line_is_not_tied(self):
        _, a, b = pitch_segment(4)
        middle = (a+b)/2
        across = camera.warp(np.array([middle-[3., 0.], middle+[3., 0.]]), fs._homography(ALFHEIM)).ravel()
        points, _ = bp.stroke_ties(ALFHEIM, [across], META, 40.)
        self.assertEqual(len(points), 0)

    def test_the_camera_is_kept_when_no_stroke_can_be_tied(self):
        model = fs._model_tree(META)
        model['meta'] = META
        window = {'strokes': np.empty((0, 4)), 'feet': np.empty((0, 2)), 'seen': np.empty((0, 2))}
        kept = bp.polish(ALFHEIM, window, np.empty((0, 2)), model, META)
        np.testing.assert_array_equal(kept, ALFHEIM)


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
