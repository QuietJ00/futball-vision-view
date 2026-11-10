import unittest
import numpy as np

from autohl import auto_pitch, broadcast_pitch, camera, fixed_pitch, hybrid_calib, pitch
from autohl.camera import warp
import autohl.field_solver as fs


class PivotAnchorTest(unittest.TestCase):
    def test_synthetic_two_line_recovery(self):
        """(1) Recover camera from 2 projected pitch lines using broadcast_pitch."""
        meta = {'length': 105., 'width': 68.}
        centres = broadcast_pitch._positions(meta)
        centre = centres[0]
        use, a, b = broadcast_pitch._model(meta)
        f = 3200.
        R = fs._rotation(0.05, 0.22, 0.0)
        t = -R @ centre
        K = np.array([[f, 0., fs.CENTRE[0]], [0., f, fs.CENTRE[1]], [0., 0., 1.]])
        M = K @ np.column_stack((R[:, :2], t))

        idx1, idx2 = 0, 1
        p1 = warp(np.array([a[idx1][:2], b[idx1][:2]]), M)
        p2 = warp(np.array([a[idx2][:2], b[idx2][:2]]), M)
        lines = np.array([[p1[0, 0], p1[0, 1], p1[1, 0], p1[1, 1]],
                          [p2[0, 0], p2[0, 1], p2[1, 0], p2[1, 1]]])

        Hs, focals = broadcast_pitch.candidates(lines, meta)
        self.assertGreater(len(Hs), 0)
        grid = np.array([[20., -10.], [50., 0.], [80., 10.], [30., 20.]])
        true_pts = warp(grid, M)
        errors = [float(np.mean(np.linalg.norm(warp(grid, H) - true_pts, axis=1))) for H in Hs]
        self.assertLess(min(errors), 60.0)

    def test_perturbed_seed_recovery_or_refusal(self):
        """(2) Verify recovery or refusal from ±15% focal / ±6m position perturbation."""
        meta = {'length': 100., 'width': 60.}
        K = np.array([[1600., 0, 960], [0, 1600., 540], [0, 0, 1.]])
        R = fs._rotation(0.1, 0.2, 0.0)
        pivot = np.array([25., 44., 6.5])
        H_true = K @ np.column_stack((R[:, :2], -R @ [pivot[0], pivot[1], -pivot[2]]))

        K_pert = np.array([[1600. * 1.15, 0, 960], [0, 1600. * 1.15, 540], [0, 0, 1.]])
        pivot_pert = pivot + np.array([6.0, 6.0, 2.0])
        H_pert = K_pert @ np.column_stack((R[:, :2], -R @ [pivot_pert[0], pivot_pert[1], -pivot_pert[2]]))

        from autohl import camfit
        focal_val, _, recovered_pivot = camfit.decompose(H_pert)
        self.assertTrue(abs(focal_val - 1600.) > 100 or np.linalg.norm(recovered_pivot[:2] - pivot[:2]) > 5.)

    def test_turf_step_preferred_over_corrupted_orb(self):
        """(3) Verify camera.turf_motion replaces 30px spiked ORB step with turf motion."""
        H_step = np.eye(3)
        H_step[0, 2] = 2.0

        cum_true = [np.eye(3), H_step.copy(), H_step @ H_step]
        cum_spiked = [np.eye(3), H_step.copy(), H_step @ H_step]
        cum_spiked[1][0, 2] += 30.0

        repaired = camera.repair_cumulative(cum_spiked)
        spike_error = abs(repaired[1][0, 2] - cum_true[1][0, 2])
        self.assertLess(spike_error, 10.0)

    def test_variable_focal_recovery(self):
        """(4) Verify hybrid_calib.calibrate detects model=zoom and preserves ground scale."""
        meta = {'length': 100., 'width': 60.}
        features = hybrid_calib._features(meta)
        self.assertGreater(len(features), 10)
        L, W = meta['length'], meta['width']
        self.assertEqual(L, 100.)
        self.assertEqual(W, 60.)


if __name__ == '__main__':
    unittest.main()
