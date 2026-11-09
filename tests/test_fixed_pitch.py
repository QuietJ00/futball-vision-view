import unittest
from argparse import Namespace

import numpy as np

from autohl import auto_pitch, field_solver, fixed_pitch, pitch
from autohl.pitch_replay import _venue


class FixedPitchTest(unittest.TestCase):
    def test_custom_dimensions_and_offset_pan(self):
        meta = _venue(Namespace(venue=None, length=108., width=66., box_depth=16.2),
                      {'venue': 'noksi'})
        self.assertEqual((meta['length'], meta['width'], meta['box_depth']), (108., 66., 16.2))
        K = np.array([[1500., 0, 960], [0, 1500., 540], [0, 0, 1.]])
        rotations = np.array([field_solver._rotation(yaw, .18, .01)
                              for yaw in np.linspace(-.44, .35, 15)])
        pivot = np.array([30., 48., -9.])
        offset = .18
        centres = pivot + offset * rotations[:, 2, :]
        self.assertGreater(np.linalg.norm(centres[0] - centres[-1]), .1)
        truth = np.array([K @ R @ np.column_stack((np.eye(3)[:, :2], -C))
                          for R, C in zip(rotations, centres)])
        motion = truth @ np.linalg.inv(truth[0])
        lines = pitch.model_lines(meta)

        def marks(frames):
            return {k: {i: uv[inside][::5].tolist()
                        for i, line in enumerate(lines)
                        for uv in [fixed_pitch._map(truth[k], line)]
                        for inside in [((uv >= 0) & (uv < [1920, 1080])).all(axis=1)]
                        if inside.sum() >= 12} for k in frames}

        calib = fixed_pitch.fit(motion, np.full(15, 100), meta, marks([0, 7, 14]), offset)
        error = fixed_pitch.errors(calib, marks([3, 11]))
        self.assertLess(max(e['p90_px'] for lines in error.values() for e in lines.values()), 2.)
        np.testing.assert_allclose(calib.pivot, pivot*[1, 1, -1], atol=.3)
        pixels = calib.to_img(3, [[35., 6.]])
        np.testing.assert_allclose(calib.to_pitch(3, pixels), [[35., 6.]], atol=1e-8)
        self.assertEqual(calib.lens_offset_m, offset)

    def test_failed_check_and_camera_break_withhold_positions(self):
        class Calibration:
            matrices = np.empty((12, 3, 3))
            same_segment = np.arange(12) < 10

        checks = {3: {0: {'p90_px': 3.}, 2: {'p90_px': 5.}},
                  7: {0: {'p90_px': 3.}, 2: {'p90_px': 15.}},
                  10: {0: {'p90_px': 3.}, 2: {'p90_px': 5.}}}
        valid = fixed_pitch.supported_frames(Calibration(), checks, 10., 2)
        self.assertEqual(np.flatnonzero(valid).tolist(), [1, 2, 3, 4, 5])
        valid = fixed_pitch.supported_frames(Calibration(), checks, 10., 2, {5: 12.})
        self.assertEqual(np.flatnonzero(valid).tolist(), [1, 2])

    def test_far_touchline_surface_check_is_one_sided(self):
        calib = fixed_pitch.Calibration(
            {'width': 60.}, 0, np.zeros(3), 0., 1000.,
            np.array([[[1., 0., 0.], [0., 1., 100.], [0., 0., 1.]]]),
            np.ones(1, dtype=bool))
        self.assertEqual(fixed_pitch.far_edge_errors(calib, {0: [[10, 90], [60, 90], [110, 90]]}),
                         {0: 20.})
        self.assertEqual(fixed_pitch.far_edge_errors(calib, {0: [[10, 60], [60, 60], [110, 60]]}),
                         {0: 0.})

    def test_consistent_rejects_tail_jump_and_handles_periodic_circle(self):
        x = np.linspace(100., 800., 28)
        y = 0.25 * x + 120.
        y[-4:] += 9.5  # Adjacent futsal line jump at tail
        pts = np.c_[x, y]
        shifts = np.zeros(len(x))
        shifts[-4:] = 9.5
        idx = np.arange(len(x))
        mask = auto_pitch._consistent(pts, shifts, idx, is_line=True)
        self.assertTrue(mask[:-4].all())
        self.assertFalse(mask[-4:].any())

        # Periodic circle seam (indices wrapping around 0 and total_samples-1)
        total = 120
        s_idx = np.r_[np.arange(0, 15), np.arange(105, 120)]
        theta = 2. * np.pi * s_idx / (total - 1)
        normals = np.c_[np.cos(theta), np.sin(theta)]
        curve_shifts = 3.0 * np.cos(theta) - 2.0 * np.sin(theta) + 1.0
        curve_shifts[5] += 18.0  # Outlier
        c_mask = auto_pitch._consistent(np.zeros((len(s_idx), 2)), curve_shifts, s_idx,
                                        is_line=False, normal=normals, total_samples=total)
        self.assertFalse(c_mask[5])
        self.assertGreaterEqual(c_mask.sum(), len(s_idx) - 2)

    def test_repair_cumulative_preserves_fast_pan(self):
        from autohl import camera
        focal, centre = 1500., np.array([960., 540.])
        K = np.array([[focal, 0, centre[0]], [0, focal, centre[1]], [0, 0, 1.]])
        def rotation(yaw):
            c, s = np.cos(yaw), np.sin(yaw)
            return K @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ np.linalg.inv(K)
        pan = np.array([rotation(.08*k) for k in range(15)])
        points = np.array([[960., 540.], [200., 900.], [1700., 200.]])
        repaired = camera.repair_cumulative(pan)
        fast_pan_error = max(float(np.linalg.norm(camera.warp(points, a)-camera.warp(points, b), axis=1).max())
                             for a, b in zip(pan, repaired))
        self.assertLessEqual(fast_pan_error, 1.0)

    def test_repair_cumulative_removes_single_frame_spike(self):
        from autohl import camera
        focal, centre = 1500., np.array([960., 540.])
        K = np.array([[focal, 0, centre[0]], [0, focal, centre[1]], [0, 0, 1.]])
        def rotation(yaw):
            c, s = np.cos(yaw), np.sin(yaw)
            return K @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ np.linalg.inv(K)
        still = np.array([rotation(.002*k) for k in range(15)])
        spiked = still.copy()
        shift = np.eye(3)
        shift[1, 2] = 46.
        spiked[7] = shift @ still[7]
        points = np.array([[960., 540.], [200., 900.], [1700., 200.]])
        fixed = camera.repair_cumulative(spiked)
        spike_error = float(np.linalg.norm(camera.warp(points, fixed[7])-camera.warp(points, still[7]), axis=1).max())
        self.assertLessEqual(spike_error, 2.0)

    def test_repair_cumulative_focal_length_invariance(self):
        from autohl import camera
        points = np.array([[960., 540.], [200., 900.], [1700., 200.]])
        for f in (1000, 1550, 2170, 5400):
            K = np.array([[f, 0, 960.], [0, f, 540.], [0, 0, 1.]])
            def rot(yaw):
                c, s = np.cos(yaw), np.sin(yaw)
                return K @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ np.linalg.inv(K)
            clean = np.array([rot(.01 * k) for k in range(30)])
            spiked = clean.copy()
            spiked[14] = np.array([[1, 0, 0], [0, 1, 46.], [0, 0, 1]]) @ clean[14]
            repaired = camera.repair_cumulative(spiked)
            residual = float(np.linalg.norm(camera.warp(points, repaired[14]) - camera.warp(points, clean[14]), axis=1).max())
            self.assertLessEqual(residual, 5.0)

    def test_supported_frames_rejects_injected_error(self):
        class Calibration:
            matrices = np.empty((30, 3, 3))
            same_segment = np.ones(30, dtype=bool)
        # Injected error: all checking frames fail threshold_px=10.0
        checks = {k: {0: {'p90_px': 14.5}, 1: {'p90_px': 18.2}} for k in range(2, 28, 4)}
        valid = fixed_pitch.supported_frames(Calibration(), checks, 10., 15)
        self.assertEqual(int(valid.sum()), 0)

    def test_supported_frames_rejects_wrong_seed_low_pass_rate(self):
        class Calibration:
            matrices = np.empty((60, 3, 3))
            same_segment = np.ones(60, dtype=bool)
        # 10 checking frames, only 2 pass (20% < 50% pass rate gate)
        checks = {k: {0: {'p90_px': 4.0 if k in (10, 20) else 15.0},
                      1: {'p90_px': 4.0 if k in (10, 20) else 16.0}} for k in range(5, 55, 5)}
        valid = fixed_pitch.supported_frames(Calibration(), checks, 10., 15)
        self.assertEqual(int(valid.sum()), 0)


if __name__ == '__main__':
    unittest.main()
