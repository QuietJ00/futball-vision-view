import unittest

import numpy as np

from score import Projection, gates, paint_scores
from sensor import aligned, frame_times, seconds


class BenchmarkTests(unittest.TestCase):
    def test_segment_times_follow_each_native_start(self):
        first = '2013-11-07 21:05:50.587623'
        second = '2013-11-07 21:05:53.588407'
        clip = {'sources': [f'0000_{first}.h264', f'0001_{second}.h264'], 'segment_seconds': 3}
        t = frame_times(clip, 30, 180)
        self.assertEqual(t[0], seconds(first))
        self.assertEqual(t[90], seconds(second))
        self.assertAlmostEqual(t[-1] - t[90], 89/30, places=5)
        with self.assertRaises(ValueError):
            frame_times(clip, 30, 179)

    def test_speed_samples_with_large_gaps_are_ineligible(self):
        samples = np.array([[0., 0., 0., 10.], [.1, 2., 0., 12.], [1., 3., 0., 20.]])
        result = aligned(samples, [-.01, .05, .5, 1.01])
        np.testing.assert_allclose(result[1], [1., 0., 11.])
        self.assertTrue(np.isnan(result[[0, 2, 3]]).all())

    def test_abstention_reduces_accuracy_yield(self):
        C = Projection(dict(matrices=np.repeat(np.eye(3)[None], 2, axis=0),
                            valid=np.array([True, False]), segments=np.zeros(2), k1=0., focal=1.))
        labels = {'0': {'lines': {'4': [[50, 5], [50, -10]]}},
                  '1': {'lines': {'4': [[50, 5], [50, -10]]}}}
        paint = paint_scores(C, {'length': 100., 'width': 60.}, labels)
        self.assertEqual(paint['scored_frames'], 1)
        self.assertEqual(paint['labelled_frames'], 2)
        row = {'error': None, 'supported': 1, 'frames': 2, 'paint': paint}
        checks = gates({'class': 'elevated_fixed'}, row, {'elevated_frame_coverage': .9,
                       'elevated_good_labelled_share': .8, 'line_p95_px': 20})
        self.assertFalse(checks['field_accuracy_yield'])

    def test_crash_is_not_a_refusal(self):
        row = {'error': 'ValueError', 'supported': 0}
        self.assertFalse(gates({'class': 'negative_unmarked'}, row, {})['no_false_accept'])


if __name__ == '__main__':
    unittest.main()
