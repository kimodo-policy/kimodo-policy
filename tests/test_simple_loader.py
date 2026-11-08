import json
import pickle
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from data.datasetloader import MultiSourceG1Dataset
from data.domain_randomization import build_domain_randomization
from data.simple_hand import SIMPLE_RIGHT_HAND_CLOSE
from data.simple_loader import SimpleAdapter, reference_root_from_source_action, simple_training_arrays
from motion.g1_reference import CANONICAL_G1_JOINT_NAMES_29, UNITREE_G1_JOINT_NAMES_29


def recorded_episode(length=4):
    command = np.zeros((length, 36), dtype=np.float32)
    command[:, 31] = 0.75
    command[:, 7:14] = 0.75 * SIMPLE_RIGHT_HAND_CLOSE
    command[:, 14:28] = np.arange(14, dtype=np.float32) / 100
    command[:, 28:31] = [0.1, 0.2, 0.3]
    hand = np.zeros((length, 14), dtype=np.float32)
    hand[:, 7:14] = 0.25 * SIMPLE_RIGHT_HAND_CLOSE
    return {"observation.leg_joints": np.full((length, 15), 0.01, np.float32),
            "observation.arm_joints": np.full((length, 14), 0.02, np.float32),
            "observation.hand_joints": hand, "action": command,
            "task_index": np.zeros(length, dtype=np.int64)}


def write_official_task(root, indices=(7,), task_name="G1WholebodyBendPickMP-v0", length=4):
    task = root / task_name
    (task / "meta").mkdir(parents=True)
    info = {"codebase_version": "v2.1", "fps": 50, "chunks_size": 5,
            "total_episodes": len(indices),
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4",
            "features": {"observation.images.egocentric": {"dtype": "video"}}}
    (task / "meta/info.json").write_text(json.dumps(info))
    (task / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "Pick up the object."}) + "\n")
    metadata = []
    for index in indices:
        values = dict(episode_index=index, episode_chunk=index // 5)
        path = task / info["data_path"].format(**values)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({k: v.tolist() for k, v in recorded_episode(length).items()}), path)
        video = task / info["video_path"].format(**values)
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"fixture: discovery only")
        metadata.append({"episode_index": index, "length": length, "tasks": [0],
                         "dataset_from_index": 800 + index * length,
                         "dataset_to_index": 799 + (index + 1) * length})
    (task / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metadata))
    return task


class SimpleAdapterTest(unittest.TestCase):
    def test_release_audit_keeps_terminal_padding_and_rejects_nan_commands(self):
        from utils import check_datasets as audit
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = write_official_task(root, length=60)
            adapter = SimpleAdapter(root, {"task": task.name, "hand_control_mode": "continuous"}, 30, 50)
            episode = adapter.episodes[0]
            audit._initialize_worker({"Simple": {"hand_control_mode": "continuous"}}, 50)
            self.assertEqual(audit._audit_episode(("Simple", episode))["status"], "valid")
            self.assertEqual(audit._post_trim_windows(episode, 0), episode.sample_count)
            values = recorded_episode(60)
            values["action"][2, 31] = np.nan
            pq.write_table(pa.table({k: v.tolist() for k, v in values.items()}), episode.data_path)
            audit._initialize_worker({"Simple": {"hand_control_mode": "continuous"}}, 50)
            self.assertEqual(audit._audit_episode(("Simple", episode))["status"], "error")

    def test_official_layout_needs_no_export_and_ignores_global_inclusive_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = write_official_task(root, indices=(7, 12), length=60)
            adapter = SimpleAdapter(root, {"task": task.name, "hand_control_mode": "continuous"}, 30, 2)
            self.assertEqual(len(adapter.episodes), 2)
            episode = adapter.episodes[0]
            self.assertEqual(episode.instruction, "Pick up the object.")
            self.assertEqual(episode.metadata["row_start"], 0)
            self.assertEqual(episode.metadata["row_end"], 60)
            self.assertEqual(episode.target_length, 36)
            self.assertEqual(episode.sample_count, 36)
            self.assertIn("chunk-001", str(episode.data_path))
            self.assertIn("chunk-002", str(adapter.episodes[1].data_path))
            self.assertEqual(type(pickle.loads(pickle.dumps(adapter))), SimpleAdapter)
            loaded = adapter.load_episode(episode)
            self.assertEqual(loaded["target_motion"].shape, (37, 417))
            self.assertTrue(torch.isfinite(loaded["target_motion"]).all())
            torch.testing.assert_close(loaded["observed_hand"][:, 1], torch.full((37,), 0.25))
            torch.testing.assert_close(loaded["target_hand"][:, 1], torch.full((37,), 0.75))
            torch.testing.assert_close(loaded["target_motion"][-1], loaded["target_motion"][-2])

    def test_missing_video_fails_instead_of_silently_dropping_an_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = write_official_task(root)
            next(task.rglob("*.mp4")).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing official Simple"):
                SimpleAdapter(root, {}, 30, 2)

    def test_wrong_length_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = write_official_task(root)
            path = task / "meta/episodes.jsonl"
            row = json.loads(path.read_text()); row["length"] = 3
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "actual_rows=4"):
                SimpleAdapter(root, {}, 30, 2)

    def test_task_label_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = write_official_task(root)
            path = next(task.rglob("*.parquet"))
            table = pq.read_table(path).to_pydict(); table["task_index"] = [1] * 4
            pq.write_table(pa.table(table), path)
            adapter = SimpleAdapter(root, {}, 30, 2)
            with self.assertRaisesRegex(ValueError, "task_index differs"):
                adapter.load_episode(adapter.episodes[0])

    def test_multiple_language_instructions_have_distinct_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = write_official_task(root, indices=(7, 12))
            path = task / "meta/tasks.jsonl"
            with path.open("a") as f:
                f.write(json.dumps({"task_index": 1, "task": "Pick up the bottle."}) + "\n")
            path = task / "meta/episodes.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["tasks"] = [1]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            adapter = SimpleAdapter(root, {}, 30, 2)
            self.assertNotEqual(adapter.episodes[0].task_id, adapter.episodes[1].task_id)
            self.assertEqual(adapter.episodes[1].instruction, "Pick up the bottle.")

    def test_selection_and_legacy_camera_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_official_task(root)
            write_official_task(root, task_name="G1WholebodyOpenFaucetTeleop-v0")
            adapter = SimpleAdapter(root, {"tasks": ["*MP-v0"], "camera": "observation.images.front"}, 30, 2)
            self.assertEqual(len(adapter.episodes), 1)
            with self.assertRaisesRegex(ValueError, "either task or tasks"):
                SimpleAdapter(root, {"task": "*", "tasks": ["*"]}, 30, 2)

    def test_continuous_preserves_commands_and_measured_observations(self):
        table = recorded_episode()
        arrays = simple_training_arrays(table, 50, "continuous")
        canonical = {name: i for i, name in enumerate(CANONICAL_G1_JOINT_NAMES_29)}
        target = arrays["target_actions"]
        np.testing.assert_allclose(arrays["observed_hand"], np.tile([0, 0.25], (4, 1)))
        np.testing.assert_allclose(arrays["target_hand"], np.tile([0, 0.75], (4, 1)))
        for name in UNITREE_G1_JOINT_NAMES_29[:12]:
            np.testing.assert_allclose(target[:, 9 + canonical[name]], 0.01)
        for offset, name in enumerate(UNITREE_G1_JOINT_NAMES_29[15:]):
            np.testing.assert_allclose(target[:, 9 + canonical[name]], table["action"][:, 14 + offset])
            np.testing.assert_allclose(arrays["observed_body"][:, canonical[name]], 0.02)
        for name, value in (("waist_yaw_joint", 0.3), ("waist_roll_joint", 0.1), ("waist_pitch_joint", 0.2)):
            np.testing.assert_allclose(target[:, 9 + canonical[name]], value)

    def test_binary_keeps_recorded_body_and_threshold_hand_targets(self):
        table = recorded_episode()
        arrays = simple_training_arrays(table, 50, "binary")
        np.testing.assert_array_equal(arrays["target_actions"][:, 9:38], arrays["observed_body"])
        np.testing.assert_array_equal(arrays["target_hand"], np.tile([0, 1], (4, 1)))
        np.testing.assert_array_equal(arrays["observed_hand"], arrays["target_hand"])

    def test_joint_reordering(self):
        table = recorded_episode()
        table["observation.leg_joints"][:] = np.arange(15)
        table["observation.arm_joints"][:] = np.arange(15, 29)
        arrays = simple_training_arrays(table, 50, "binary")
        for i, name in enumerate(CANONICAL_G1_JOINT_NAMES_29):
            np.testing.assert_array_equal(arrays["observed_body"][:, i], np.full(4, UNITREE_G1_JOINT_NAMES_29.index(name)))

    def test_root_heading_wrap_and_velocity_units(self):
        command = recorded_episode()["action"]
        command[:, 32:34] = [1, -0.5]
        command[:, 35] = [3.1, -3.1, -3, -2.9]
        result = reference_root_from_source_action(command, 50)
        np.testing.assert_array_equal(result["local_xy_delta"][0], [0, 0])
        np.testing.assert_allclose(result["local_xy_delta"][1:], np.tile([0.02, -0.01], (3, 1)))
        np.testing.assert_allclose(result["rotation_matrices"][0], np.eye(3))
        self.assertAlmostEqual(float(np.arctan2(result["rotation_matrices"][1, 1, 0], result["rotation_matrices"][1, 0, 0])), 2 * np.pi - 6.2, places=6)

    def test_nonfinite_and_misaligned_data_are_rejected(self):
        table = recorded_episode()
        table["action"][2, 20] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            simple_training_arrays(table, 50, "continuous")
        table = recorded_episode(); table["observation.arm_joints"] = table["observation.arm_joints"][:-1]
        with self.assertRaisesRegex(ValueError, "lengths"):
            simple_training_arrays(table, 50, "continuous")


class DomainRandomizationDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = write_official_task(self.root, length=60)
        self.image = torch.arange(3 * 16 * 24).remainder(256).to(torch.uint8).reshape(3, 16, 24)

    @staticmethod
    def config(**overrides):
        return {
            "enabled": True,
            "color_jitter": {
                "probability": 0.8, "brightness": 0.2, "contrast": 0.2,
                "saturation": 0.2, "hue": 0.02, **overrides,
            },
        }

    def dataset(self, *, training=True, augmentation=None):
        selection = {"task": self.task.name, "hand_control_mode": "continuous"}
        dataset = MultiSourceG1Dataset(
            dataset_roots={"Simple": str(self.root)},
            dataset_selection={"Simple": selection},
            action_history=2,
            action_chunk=2,
            sampling_seed=42,
            training=training,
            domain_randomization=augmentation,
        )
        self.addCleanup(dataset.close)
        reader = patch.object(dataset, "_read_video_frame", return_value=self.image)
        reader.start()
        self.addCleanup(reader.stop)
        return dataset

    def test_only_image_changes_and_sampling_and_all_labels_are_identical(self):
        original = self.dataset()
        augmented = self.dataset(augmentation=self.config(probability=1.0))
        untouched_image = self.image.clone()
        for index in range(6):
            before, after = original[index], augmented[index]
            self.assertFalse(torch.equal(before["egoview"], after["egoview"]))
            self.assertEqual(after["egoview"].shape, self.image.shape)
            self.assertEqual(after["egoview"].dtype, torch.uint8)
            self.assertTrue(after["egoview"].is_contiguous())
            self.assertEqual(before.keys(), after.keys())
            for key in before:
                if key == "egoview":
                    continue
                if isinstance(before[key], torch.Tensor):
                    self.assertTrue(torch.equal(before[key], after[key]), key)
                else:
                    self.assertEqual(before[key], after[key], key)
        self.assertTrue(torch.equal(self.image, untouched_image))

    def test_default_disabled_and_nontraining_images_remain_identical(self):
        crop_config = {"enabled": True, "view_crop": {
            "enabled": True, "probability": 1.0, "min_scale": 0.95,
        }}
        for training, augmentation in (
            (True, None),
            (True, {}),
            (True, {"color_jitter": self.config()["color_jitter"]}),
            (True, {"enabled": False}),
            (False, self.config(probability=1.0)),
            (False, crop_config),
            (True, self.config(probability=0.0)),
        ):
            with self.subTest(training=training, augmentation=augmentation):
                dataset = self.dataset(training=training, augmentation=augmentation)
                self.assertTrue(torch.equal(dataset[0]["egoview"], self.image))

        # Dataset callers outside train.py are safe even with the enabled config.
        dataset = MultiSourceG1Dataset(
            dataset_roots={"Simple": str(self.root)},
            dataset_selection={"Simple": {}},
            domain_randomization=self.config(),
            action_chunk=2,
        )
        self.addCleanup(dataset.close)
        self.assertFalse(dataset.training)
        self.assertIsNone(dataset._domain_randomization)

    def test_view_crop_preserves_realworld_sampling_and_all_nonimage_fields(self):
        dataset = self.dataset(augmentation={"enabled": True, "view_crop": {
            "enabled": True, "probability": 1.0, "min_scale": 0.7,
        }})
        adapter, episode, cut = dataset._sample_record(dataset._rng_for_index(0))
        with patch.object(dataset, "_sample_record", return_value=(
            adapter, replace(episode, source="RealWorld"), cut
        )):
            after = dataset[0]
            dataset.training = False
            before = dataset[0]
        self.assertFalse(torch.equal(before["egoview"], after["egoview"]))
        for key in before:
            if key == "egoview":
                continue
            if isinstance(before[key], torch.Tensor):
                self.assertTrue(torch.equal(before[key], after[key]), key)
            else:
                self.assertEqual(before[key], after[key], key)

    def test_every_source_changes_images_only_with_explicit_opt_in(self):
        for enabled in (False, True):
            dataset = self.dataset(augmentation=self.config(probability=1.0) if enabled else None)
            adapter, episode, cut = dataset._sample_record(dataset._rng_for_index(0))
            for source in ("Simple", "HumanoidArena", "RealWorld", "HIW500", "HumanoidEveryday", "UnifoLM_WBT_Dataset"):
                with self.subTest(source=source, enabled=enabled), patch.object(
                    dataset, "_sample_record", return_value=(adapter, replace(episode, source=source), cut)
                ):
                    unchanged = torch.equal(dataset[0]["egoview"], self.image)
                    self.assertEqual(unchanged, not enabled)

    def test_reproducible_after_pickle_without_changing_global_rngs(self):
        augment = build_domain_randomization(self.config(probability=1.0))
        python_state = random.getstate()
        torch_state = torch.get_rng_state().clone()
        result = augment(self.image, seed=42, index=17)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        restored = pickle.loads(pickle.dumps(augment))
        self.assertTrue(torch.equal(result, restored(self.image, seed=42, index=17)))
        self.assertFalse(torch.equal(result, augment(self.image, seed=42, index=18)))
        self.assertFalse(torch.equal(result, augment(self.image, seed=43, index=17)))

    def test_probability_keeps_a_mix_of_original_and_augmented_frames(self):
        augment = build_domain_randomization(self.config())
        changed = [not torch.equal(self.image, augment(self.image, seed=42, index=i)) for i in range(40)]
        self.assertTrue(any(changed))
        self.assertFalse(all(changed))

    def test_invalid_strengths_fail_early(self):
        for config in ({"probability": 1.1}, {"hue": 0.6}, {"brightness": -0.1}, {"contrast": float("nan")}, {"saturation": float("inf")}, {"hue": None}, {"probability": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.dataset(augmentation=self.config(**config))

    def test_enabled_requires_all_parameters_and_rejects_typos(self):
        for name in self.config()["color_jitter"]:
            config = self.config()
            del config["color_jitter"][name]
            with self.subTest(missing=name), self.assertRaisesRegex(ValueError, name):
                self.dataset(augmentation=config)
        for config in (
            {"enabled": True},
            {"enabled": True, "colour_jitter": {}},
            self.config(brigthness=0.2),
            {"enabled": "false"},
        ):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "domain_randomization"):
                self.dataset(augmentation=config)

    def test_zero_strengths_leave_pixels_exactly_unchanged(self):
        augment = build_domain_randomization(self.config(
            probability=1.0, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0
        ))
        self.assertTrue(torch.equal(augment(self.image, seed=42, index=0), self.image))


if __name__ == "__main__":
    unittest.main()
