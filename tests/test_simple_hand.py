import unittest
from types import SimpleNamespace

import numpy as np
import torch

from data.simple_hand import (
    SIMPLE_LEFT_HAND_CLOSE,
    SIMPLE_RIGHT_HAND_CLOSE,
    hand_targets_from_closure,
    mjcf_hand_to_wbc,
    project_hand_closure,
    source_action_hand_targets,
)
from motion.g1_reference import (
    resample_hand_continuous,
    resample_hand_continuous_chunk,
)
from evaluation.simple_server import (
    BODY_NAMES, LEFT_HAND_NAMES, RIGHT_HAND_NAMES, _robot_proprio,
    measured_hand_closure_from_proprio,
)


class SimpleHandTest(unittest.TestCase):
    def test_legacy_mp_joint_observations_preserve_measured_closure(self):
        hands = hand_targets_from_closure(np.asarray([[0.25, 0.75]], np.float32))[0]
        joints = {name: SimpleNamespace(qpos=[0.0], qvel=[0.0]) for name in BODY_NAMES}
        joints.update({name: SimpleNamespace(qpos=[value]) for name, value in
                       zip(LEFT_HAND_NAMES + RIGHT_HAND_NAMES, hands)})
        actual = measured_hand_closure_from_proprio(_robot_proprio(SimpleNamespace(joints=joints)))
        np.testing.assert_allclose(actual, [0.25, 0.75], atol=1e-6)

    def test_projection_round_trip(self):
        closure = np.asarray([[0.0, 1.0], [0.25, 0.75]], dtype=np.float32)
        targets = hand_targets_from_closure(closure)
        np.testing.assert_allclose(project_hand_closure(targets), closure)

    def test_source_action_reorders_left_hand_before_projection(self):
        source = np.zeros((1, 36), dtype=np.float32)
        source[0, :7] = np.asarray(
            [
                *SIMPLE_LEFT_HAND_CLOSE[:3],
                *SIMPLE_LEFT_HAND_CLOSE[5:7],
                *SIMPLE_LEFT_HAND_CLOSE[3:5],
            ],
            dtype=np.float32,
        )
        source[0, 7:14] = SIMPLE_RIGHT_HAND_CLOSE
        np.testing.assert_allclose(
            project_hand_closure(source_action_hand_targets(source)),
            np.ones((1, 2), dtype=np.float32),
        )

    def test_mjcf_observation_reorders_both_hands_before_projection(self):
        natural = hand_targets_from_closure(
            np.asarray([[0.25, 0.75]], dtype=np.float32)
        )
        mjcf = np.concatenate(
            (
                natural[:, :3],
                natural[:, 5:7],
                natural[:, 3:5],
                natural[:, 7:10],
                natural[:, 12:14],
                natural[:, 10:12],
            ),
            axis=1,
        )
        np.testing.assert_allclose(
            project_hand_closure(mjcf_hand_to_wbc(mjcf)),
            np.asarray([[0.25, 0.75]], dtype=np.float32),
        )

    def test_proprio_hand_closure_uses_mjcf_observation_order(self):
        natural = hand_targets_from_closure(
            np.asarray([[0.25, 0.75]], dtype=np.float32)
        )[0]
        proprio = {
            "left_hand_q": np.concatenate(
                (natural[:3], natural[5:7], natural[3:5])
            ),
            "right_hand_q": np.concatenate(
                (natural[7:10], natural[12:14], natural[10:12])
            ),
        }
        np.testing.assert_allclose(
            measured_hand_closure_from_proprio(proprio),
            np.asarray([0.25, 0.75], dtype=np.float32),
        )

    def test_linear_chunk_resampling_preserves_previous_boundary(self):
        source = torch.tensor([[0.5, 0.0], [1.0, 1.0]])
        actual = resample_hand_continuous_chunk(
            source,
            source_fps=1,
            target_fps=2,
            previous_hand_closure=torch.tensor([0.0, 0.0]),
        )
        expected = torch.tensor(
            [[0.25, 0.0], [0.5, 0.0], [0.75, 0.5], [1.0, 1.0]]
        )
        torch.testing.assert_close(actual, expected)
        resampled = resample_hand_continuous(source, source_fps=1, target_fps=2)
        torch.testing.assert_close(
            resampled,
            torch.tensor([[0.5, 0.0], [0.75, 0.5], [1.0, 1.0]]),
        )

    def test_projection_rejects_non_rank_one_target_when_requested(self):
        target = hand_targets_from_closure(np.asarray([[1.0, 1.0]], dtype=np.float32))
        target[0, 0] += 0.5
        with self.assertRaisesRegex(ValueError, "not representable"):
            project_hand_closure(target, max_relative_residual=0.01)


if __name__ == "__main__":
    unittest.main()
