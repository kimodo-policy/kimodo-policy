import unittest
import numpy as np
from hand_stability import stabilize_binary_hand as filt


class HandStabilityTests(unittest.TestCase):
    def test_threshold_jitter_does_not_toggle(self):
        p = np.tile([[.49, .51], [.51, .49]], (40, 1))
        out, _ = filt(p, [0, 1])
        np.testing.assert_equal(out, np.tile([0, 1], (80, 1)))

    def test_brief_open_pulse_preserves_grasp_but_sustained_open_releases(self):
        p = np.ones((60, 2))
        p[2:5, 0] = 0
        p[12:40, 0] = 0
        out, _ = filt(p, [1, 1])
        self.assertTrue(np.all(out[:32, 0] == 1))
        self.assertEqual(out[32, 0], 0)
        self.assertTrue(np.all(out[:, 1] == 1))

    def test_confirmation_crosses_accepted_chunk_boundary(self):
        a, state = filt(np.ones((4, 2)), [0, 0])
        b, _ = filt(np.ones((3, 2)), a[-1], state=state)
        np.testing.assert_equal(b[:, 0], [0, 1, 1])

    def test_release_confirmation_longer_than_execution_chunk(self):
        a, state = filt(np.zeros((15, 2)), [1, 1])
        self.assertTrue(np.all(a == 1))
        b, _ = filt(np.zeros((15, 2)), a[-1], state=state)
        self.assertTrue(np.all(b[:5] == 1))
        self.assertTrue(np.all(b[5:] == 0))

    def test_changed_boundary_discards_incompatible_pending_state(self):
        _, state = filt(np.zeros((20, 2)), [1, 1])
        b, _ = filt(np.ones((2, 2)), [0, 0], state=state)
        self.assertTrue(np.all(b == 0))

    def test_rejected_candidate_does_not_advance_confirmation(self):
        _, discarded = filt(np.ones((5, 2)), [0, 0])
        out, _ = filt(np.ones((2, 2)), [0, 0], state=None)
        self.assertTrue(np.all(out == 0))

    def test_no_input_mutation_and_hands_independent(self):
        p = np.tile([1., 0.], (30, 1))
        saved = p.copy()
        out, _ = filt(p, [0, 1])
        np.testing.assert_equal(p, saved)
        np.testing.assert_equal(out[-1], [1, 0])
        self.assertEqual(out[5, 0], 1)
        self.assertEqual(out[20, 1], 0)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            filt([[float("nan"), 0]], [0, 0])
        with self.assertRaises(ValueError):
            filt([[1, 0]], [0, 0], close_threshold=.3, open_threshold=.7)


if __name__ == "__main__":
    unittest.main()
