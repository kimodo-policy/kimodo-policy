import tempfile
import unittest
import random
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import BatchSampler, get_worker_info
from torch.optim.lr_scheduler import LambdaLR
from accelerate.data_loader import BatchSamplerShard
from omegaconf import OmegaConf

from train import (
    ResumableOrdinalSampler,
    _capture_rng_state,
    _gradient_clip_statistics,
    _load_text_embedding_cache,
    _load_rank_rng_state,
    _load_model_initialization_checkpoint,
    _load_training_checkpoint,
    _parse_args,
    _process_seed,
    _restore_rng_state,
    _rng_state_path,
    _sample_ordinal_range,
    _save_rank_rng_state,
    _save_task_text_embedding,
    _task_text_embedding_cache_path,
    _validate_init_checkpoint_config,
    _validate_resume_config,
    _worker_seed,
    build_dataset,
    build_dataloader,
)


class _TinyTrainableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = nn.Linear(3, 2)
        self.frozen = nn.Linear(3, 2)
        self.frozen.requires_grad_(False)


class _TinyDataset:
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return index


class _WorkerIdentityDataset:
    windows_per_episode = 4

    def __len__(self):
        return 64

    def __getitem__(self, index):
        worker = get_worker_info()
        return index, -1 if worker is None else worker.id


class TrainingResumeTest(unittest.TestCase):
    @staticmethod
    def _stochastic_train_step(model, optimizer, scheduler):
        optimizer.zero_grad()
        inputs = torch.rand(5, 3)
        random_scale = random.random() + float(np.random.random())
        loss = model.trainable(inputs).square().mean() * random_scale
        loss.backward()
        optimizer.step()
        scheduler.step()

    def test_python_numpy_and_torch_rng_round_trip(self):
        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(404)
        state = _capture_rng_state(rank=0, world_size=1)

        expected_python = [random.random() for _ in range(4)]
        expected_numpy = np.random.random(4)
        expected_torch = torch.rand(4)
        expected_cuda = torch.rand(4, device="cuda") if torch.cuda.is_available() else None

        for _ in range(10):
            random.random()
            np.random.random()
            torch.rand(1)
            if torch.cuda.is_available():
                torch.rand(1, device="cuda")
        _restore_rng_state(state, expected_rank=0, expected_world_size=1)

        self.assertEqual([random.random() for _ in range(4)], expected_python)
        np.testing.assert_array_equal(np.random.random(4), expected_numpy)
        torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)
        if expected_cuda is not None:
            torch.testing.assert_close(
                torch.rand(4, device="cuda"), expected_cuda, rtol=0, atol=0
            )

    def test_gradient_clip_statistics_reports_preclip_norm_and_scale(self):
        self.assertEqual(
            _gradient_clip_statistics(torch.tensor(0.25), 0.5),
            (0.25, 0.0, 1.0),
        )
        norm, was_clipped, scale = _gradient_clip_statistics(
            torch.tensor(1.0), 0.5
        )
        self.assertEqual(norm, 1.0)
        self.assertEqual(was_clipped, 1.0)
        self.assertAlmostEqual(scale, 0.5)

    def test_gradient_clip_statistics_handles_nonfinite_norm(self):
        norm, was_clipped, scale = _gradient_clip_statistics(float("inf"), 0.5)
        self.assertTrue(math.isinf(norm))
        self.assertEqual(was_clipped, 1.0)
        self.assertEqual(scale, 0.0)

    def test_gradient_clip_statistics_rejects_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            _gradient_clip_statistics(0.2, 0.0)

    def test_per_rank_rng_file_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            _save_rank_rng_state(directory, rank=2, world_size=4)

            path = Path(_rng_state_path(directory, 2))
            payload = _load_rank_rng_state(directory, rank=2, expected_world_size=4)

            self.assertTrue(path.is_file())
            self.assertEqual(payload["rank"], 2)
            self.assertEqual(payload["world_size"], 4)
            with self.assertRaisesRegex(RuntimeError, "world_size"):
                _load_rank_rng_state(directory, rank=2, expected_world_size=8)

    def test_old_checkpoint_without_rng_file_remains_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(
                _load_rank_rng_state(directory, rank=0, expected_world_size=1)
            )

    def test_new_checkpoint_rejects_missing_rank_rng_file(self):
        model = _TinyTrainableModel()
        optimizer = torch.optim.AdamW(model.trainable.parameters(), lr=0.01)
        scheduler = LambdaLR(optimizer, lambda step: 1.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint_0"
            checkpoint_path.mkdir()
            torch.save(
                {
                    "global_step": 0,
                    "model": {
                        name: parameter.detach().clone()
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    },
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "world_size": 2,
                    "rng_state_version": 1,
                },
                checkpoint_path / "training_state.pt",
            )

            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                _load_training_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    str(checkpoint_path),
                    expected_world_size=2,
                )

    def test_init_checkpoint_loads_only_model_weights_and_returns_lineage(self):
        torch.manual_seed(7)
        source_model = _TinyTrainableModel()
        with torch.no_grad():
            source_model.trainable.weight.fill_(1.25)
            source_model.trainable.bias.fill_(-0.75)

        torch.manual_seed(19)
        target_model = _TinyTrainableModel()
        target_frozen_state = {
            name: parameter.detach().clone()
            for name, parameter in target_model.frozen.named_parameters()
        }
        optimizer = torch.optim.AdamW(target_model.trainable.parameters(), lr=0.123)
        scheduler = LambdaLR(optimizer, lambda step: 0.9 ** step)
        optimizer_before = optimizer.state_dict()
        scheduler_before = scheduler.state_dict()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint_123"
            checkpoint_path.mkdir()
            torch.save(
                {
                    "global_step": 123,
                    "model": {
                        f"_orig_mod.{name}": parameter.detach().clone()
                        for name, parameter in source_model.named_parameters()
                        if parameter.requires_grad
                    },
                    "optimizer": {"must_not": "be loaded"},
                    "scheduler": {"must_not": "be loaded"},
                    "world_size": 4,
                },
                checkpoint_path / "training_state.pt",
            )
            (checkpoint_path / "config.json").write_text(
                json.dumps(
                    {
                        "initialization": {
                            "mode": "init_checkpoint",
                            "source_checkpoint": "/earlier/checkpoint_10",
                        }
                    }
                ),
                encoding="utf-8",
            )

            metadata = _load_model_initialization_checkpoint(
                target_model, str(checkpoint_path)
            )

        for name, parameter in target_model.trainable.named_parameters():
            torch.testing.assert_close(
                parameter, dict(source_model.trainable.named_parameters())[name]
            )
        for name, parameter in target_model.frozen.named_parameters():
            torch.testing.assert_close(parameter, target_frozen_state[name])
        self.assertEqual(optimizer.state_dict(), optimizer_before)
        self.assertEqual(scheduler.state_dict(), scheduler_before)
        self.assertEqual(metadata["source_global_step"], 123)
        self.assertEqual(metadata["source_world_size"], 4)
        self.assertEqual(
            metadata["source_initialization"]["source_checkpoint"],
            "/earlier/checkpoint_10",
        )

    def test_init_checkpoint_config_allows_new_data_and_optimizer(self):
        project_root = Path(__file__).resolve().parents[1]
        source_config = OmegaConf.load(project_root / "train.yaml")
        current_config = OmegaConf.create(
            OmegaConf.to_container(source_config, resolve=True)
        )
        source_config.main.dataset_selection = {
            "UnifoLM_WBT_Dataset": {"tasks": ["*"]}
        }
        current_config.main.dataset_selection = {
            "HumanoidArena": {"merged": "all_16_refpose_v3_1"}
        }
        source_config.main.max_steps = 100000
        current_config.main.max_steps = 200000
        source_config.training.optimizer.control_backbone_lr = 3.0e-5
        current_config.training.optimizer.control_backbone_lr = 1.0e-5

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory)
            (checkpoint_path / "config.json").write_text(
                json.dumps(OmegaConf.to_container(source_config, resolve=True)),
                encoding="utf-8",
            )

            _validate_init_checkpoint_config(current_config, str(checkpoint_path))
            current_config.model.controlnet_num_layers += 1
            with self.assertRaisesRegex(ValueError, "controlnet_num_layers"):
                _validate_init_checkpoint_config(current_config, str(checkpoint_path))

    def test_resume_and_init_checkpoint_cli_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            _parse_args(
                [
                    "--resume",
                    "/tmp/resume",
                    "--init-checkpoint",
                    "/tmp/init",
                ]
            )

    def test_checkpoint_rejects_inconsistent_sample_ordinal(self):
        model = _TinyTrainableModel()
        optimizer = torch.optim.AdamW(model.trainable.parameters(), lr=0.01)
        scheduler = LambdaLR(optimizer, lambda step: 1.0)
        optimizer.step()
        scheduler.step()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint_1"
            checkpoint_path.mkdir()
            torch.save(
                {
                    "global_step": 1,
                    "model": {
                        name: parameter.detach().clone()
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    },
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "world_size": 1,
                    "data_state": {
                        "sampling_version": 1,
                        "next_sample_ordinal": 7,
                        "batch_size": 4,
                        "gradient_accumulation_steps": 2,
                        "world_size": 1,
                    },
                },
                checkpoint_path / "training_state.pt",
            )

            with self.assertRaisesRegex(RuntimeError, "next_sample_ordinal"):
                _load_training_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    str(checkpoint_path),
                    expected_world_size=1,
                )

    def test_resumed_stochastic_step_matches_uninterrupted_training(self):
        random.seed(11)
        np.random.seed(22)
        torch.manual_seed(33)
        source_model = _TinyTrainableModel()
        source_optimizer = torch.optim.AdamW(
            source_model.trainable.parameters(), lr=0.01
        )
        source_scheduler = LambdaLR(source_optimizer, lambda step: 0.95 ** step)
        self._stochastic_train_step(
            source_model, source_optimizer, source_scheduler
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint_1"
            checkpoint_path.mkdir()
            _save_rank_rng_state(str(checkpoint_path), rank=0, world_size=1)
            torch.save(
                {
                    "global_step": 1,
                    "model": {
                        name: parameter.detach().clone()
                        for name, parameter in source_model.named_parameters()
                        if parameter.requires_grad
                    },
                    "optimizer": source_optimizer.state_dict(),
                    "scheduler": source_scheduler.state_dict(),
                    "world_size": 1,
                    "rng_state_version": 1,
                    "data_state": {
                        "sampling_version": 1,
                        "next_sample_ordinal": 5,
                        "batch_size": 5,
                        "gradient_accumulation_steps": 1,
                        "world_size": 1,
                    },
                },
                checkpoint_path / "training_state.pt",
            )

            self._stochastic_train_step(
                source_model, source_optimizer, source_scheduler
            )
            uninterrupted_state = {
                name: parameter.detach().clone()
                for name, parameter in source_model.named_parameters()
                if parameter.requires_grad
            }
            uninterrupted_optimizer = source_optimizer.state_dict()

            random.seed(101)
            np.random.seed(202)
            torch.manual_seed(303)
            resumed_model = _TinyTrainableModel()
            resumed_optimizer = torch.optim.AdamW(
                resumed_model.trainable.parameters(), lr=0.01
            )
            resumed_scheduler = LambdaLR(
                resumed_optimizer, lambda step: 0.95 ** step
            )
            resumed_step = _load_training_checkpoint(
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                str(checkpoint_path),
                expected_world_size=1,
            )
            resumed_rng = _load_rank_rng_state(
                str(checkpoint_path), rank=0, expected_world_size=1
            )
            _restore_rng_state(
                resumed_rng, expected_rank=0, expected_world_size=1
            )
            self._stochastic_train_step(
                resumed_model, resumed_optimizer, resumed_scheduler
            )

        self.assertEqual(resumed_step, 1)
        self.assertEqual(resumed_scheduler.last_epoch, source_scheduler.last_epoch)
        self.assertEqual(
            resumed_scheduler.get_last_lr(), source_scheduler.get_last_lr()
        )
        for name, parameter in resumed_model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(
                    parameter, uninterrupted_state[name], rtol=0, atol=0
                )
        resumed_optimizer_state = resumed_optimizer.state_dict()["state"]
        for parameter_id, expected_state in uninterrupted_optimizer["state"].items():
            for state_name, expected_value in expected_state.items():
                actual_value = resumed_optimizer_state[parameter_id][state_name]
                if torch.is_tensor(expected_value):
                    torch.testing.assert_close(
                        actual_value, expected_value, rtol=0, atol=0
                    )
                else:
                    self.assertEqual(actual_value, expected_value)

    def test_resume_ordinal_range_skips_all_consumed_global_samples(self):
        config = SimpleNamespace(
            main=SimpleNamespace(
                batch_size=4,
                max_steps=10,
                gradient=SimpleNamespace(grad_accumulation_steps=2),
            )
        )

        start, stop = _sample_ordinal_range(config, resume_step=3, world_size=4)
        sampler = ResumableOrdinalSampler(start, stop)

        self.assertEqual(start, 3 * 2 * 4 * 4)
        self.assertEqual(stop, 10 * 2 * 4 * 4)
        self.assertEqual(next(iter(sampler)), start)
        self.assertEqual(len(sampler), stop - start)
        self.assertEqual((stop - start) % (4 * 4), 0)

    def test_accelerate_rank_shards_cover_each_resumed_ordinal_once(self):
        batch_size = 3
        world_size = 4
        start = 120
        stop = 168
        rank_batches = []
        for rank in range(world_size):
            batch_sampler = BatchSampler(
                ResumableOrdinalSampler(start, stop),
                batch_size=batch_size,
                drop_last=True,
            )
            shard = BatchSamplerShard(
                batch_sampler,
                num_processes=world_size,
                process_index=rank,
                split_batches=False,
            )
            rank_batches.append(list(shard))

        reconstructed = []
        for microstep in range(len(rank_batches[0])):
            for rank in range(world_size):
                reconstructed.extend(rank_batches[rank][microstep])

        self.assertEqual(reconstructed, list(range(start, stop)))

    def test_eight_rank_sharding_keeps_four_window_groups_inside_each_batch(self):
        batch_size = 128
        world_size = 8
        group_size = 4
        start = 1024
        stop = start + batch_size * world_size * 2
        rank_batches = []
        for rank in range(world_size):
            batch_sampler = BatchSampler(
                ResumableOrdinalSampler(start, stop),
                batch_size=batch_size,
                drop_last=True,
            )
            shard = BatchSamplerShard(
                batch_sampler,
                num_processes=world_size,
                process_index=rank,
                split_batches=False,
            )
            rank_batches.append(list(shard))

        for batches in rank_batches:
            for batch in batches:
                self.assertEqual(len(batch), batch_size)
                self.assertEqual(batch[0] % group_size, 0)
                for offset in range(0, batch_size, group_size):
                    group = batch[offset : offset + group_size]
                    self.assertEqual(group, list(range(group[0], group[0] + group_size)))

    def test_dataloader_assigns_each_group_to_one_worker(self):
        config = SimpleNamespace(
            main=SimpleNamespace(
                cpu_workers_num=2,
                batch_size=8,
                seed=42,
                max_steps=1,
                gradient=SimpleNamespace(grad_accumulation_steps=1),
            )
        )
        data_loader = build_dataloader(
            config, train_dataset=_WorkerIdentityDataset()
        )

        indices, worker_ids = next(iter(data_loader))

        self.assertEqual(indices.tolist(), list(range(8)))
        for offset in range(0, 8, 4):
            self.assertEqual(
                len(set(worker_ids[offset : offset + 4].tolist())), 1
            )

    def test_text_cache_without_instruction_metadata_is_rejected(self):
        task_id = "HumanoidArena::task-id"
        task_instructions = {task_id: "Open the door."}
        task_cache_names = {task_id: "HSI_open_door"}
        with tempfile.TemporaryDirectory() as directory:
            cache_path = _task_text_embedding_cache_path(
                directory, task_id, task_cache_names[task_id]
            )
            Path(cache_path).parent.mkdir(parents=True)
            torch.save(
                {
                    "version": 3,
                    "source": "HumanoidArena",
                    "task_id": task_id,
                    "task_name": "HSI_open_door",
                    "embedding": torch.zeros(1, 4096),
                },
                cache_path,
            )

            loaded, missing = _load_text_embedding_cache(
                directory,
                task_instructions,
                task_cache_names,
                feature_dim=4096,
            )

        self.assertEqual(loaded, {})
        self.assertEqual(missing, [task_id])

    def test_natural_language_cache_keeps_task_id_lookup_key(self):
        task_id = "HumanoidArena::task-id"
        task_instructions = {task_id: "Open the door."}
        task_cache_names = {task_id: "HSI_open_door"}
        embedding = torch.randn(1, 16)
        with tempfile.TemporaryDirectory() as directory:
            _save_task_text_embedding(
                directory,
                task_id,
                task_cache_names[task_id],
                task_instructions[task_id],
                embedding,
            )

            loaded, missing = _load_text_embedding_cache(
                directory,
                task_instructions,
                task_cache_names,
                feature_dim=16,
            )

        self.assertEqual(missing, [])
        torch.testing.assert_close(loaded[task_id], embedding)

    def test_text_cache_is_grouped_by_source_and_sanitized_task_name(self):
        cache_path = Path(
            _task_text_embedding_cache_path(
                "/cache",
                "UnifoLM_WBT_Dataset::Dex1/task::Open the drawer.",
                "Dex1/task",
            )
        )

        self.assertEqual(cache_path.parent, Path("/cache/UnifoLM_WBT_Dataset"))
        self.assertRegex(cache_path.name, r"^Dex1_task__[0-9a-f]{12}\.pt$")

    def test_unreadable_per_task_cache_is_treated_as_missing(self):
        task_id = "HIW500::sweep_floor"
        task_instructions = {task_id: "Sweep the floor."}
        task_cache_names = {task_id: "sweep_floor"}
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(
                _task_text_embedding_cache_path(
                    directory, task_id, task_cache_names[task_id]
                )
            )
            cache_path.parent.mkdir(parents=True)
            cache_path.write_bytes(b"not a torch cache")

            loaded, missing = _load_text_embedding_cache(
                directory,
                task_instructions,
                task_cache_names,
                feature_dim=4096,
            )

        self.assertEqual(loaded, {})
        self.assertEqual(missing, [task_id])

    def test_restores_complete_training_state(self):
        torch.manual_seed(7)
        source_model = _TinyTrainableModel()
        source_optimizer = torch.optim.AdamW(source_model.trainable.parameters(), lr=0.01)
        source_scheduler = LambdaLR(source_optimizer, lambda step: 0.9 ** step)

        for _ in range(4):
            source_optimizer.zero_grad()
            source_model.trainable(torch.ones(2, 3)).square().mean().backward()
            source_optimizer.step()
            source_scheduler.step()

        trainable_state = {
            name: parameter.detach().clone()
            for name, parameter in source_model.named_parameters()
            if parameter.requires_grad
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint_4"
            checkpoint_path.mkdir()
            torch.save(
                {
                    "global_step": 4,
                    "model": trainable_state,
                    "optimizer": source_optimizer.state_dict(),
                    "scheduler": source_scheduler.state_dict(),
                    "world_size": 4,
                },
                checkpoint_path / "training_state.pt",
            )

            torch.manual_seed(19)
            resumed_model = _TinyTrainableModel()
            resumed_optimizer = torch.optim.AdamW(resumed_model.trainable.parameters(), lr=0.01)
            resumed_scheduler = LambdaLR(resumed_optimizer, lambda step: 0.9 ** step)
            global_step = _load_training_checkpoint(
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                str(checkpoint_path),
                expected_world_size=4,
            )

        self.assertEqual(global_step, 4)
        self.assertEqual(resumed_scheduler.last_epoch, 4)
        self.assertEqual(resumed_scheduler.get_last_lr(), source_scheduler.get_last_lr())
        for name, parameter in resumed_model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter, trainable_state[name])
        source_optimizer_state = source_optimizer.state_dict()["state"]
        resumed_optimizer_state = resumed_optimizer.state_dict()["state"]
        self.assertEqual(source_optimizer_state.keys(), resumed_optimizer_state.keys())
        for parameter_id in source_optimizer_state:
            torch.testing.assert_close(
                resumed_optimizer_state[parameter_id]["exp_avg"],
                source_optimizer_state[parameter_id]["exp_avg"],
            )

        resumed_optimizer.step()
        resumed_scheduler.step()
        self.assertEqual(resumed_scheduler.last_epoch, 5)

    def test_resume_seed_uses_global_step_offset(self):
        initial_seeds = {
            _worker_seed(42, 0, 4, 8, process_index, worker_id)
            for process_index in range(4)
            for worker_id in range(8)
        }
        resumed_seeds = {
            _worker_seed(42, 40000, 4, 8, process_index, worker_id)
            for process_index in range(4)
            for worker_id in range(8)
        }

        self.assertEqual(len(initial_seeds), 32)
        self.assertEqual(len(resumed_seeds), 32)
        self.assertTrue(initial_seeds.isdisjoint(resumed_seeds))
        self.assertEqual(_process_seed(42, 40000, 4), 160042)

    def test_dataloader_uses_spawn_and_single_batch_prefetch(self):
        config = SimpleNamespace(
            main=SimpleNamespace(
                cpu_workers_num=4,
                batch_size=16,
                seed=42,
                max_steps=100000,
                gradient=SimpleNamespace(grad_accumulation_steps=2),
            )
        )

        with patch("train.DataLoader") as data_loader:
            build_dataloader(
                config,
                train_dataset=object(),
                process_index=2,
                world_size=4,
                resume_step=60000,
            )

        kwargs = data_loader.call_args.kwargs
        self.assertEqual(kwargs["num_workers"], 4)
        self.assertEqual(kwargs["multiprocessing_context"], "spawn")
        self.assertEqual(kwargs["prefetch_factor"], 1)
        self.assertTrue(kwargs["persistent_workers"])
        self.assertIsInstance(kwargs["sampler"], ResumableOrdinalSampler)
        self.assertEqual(kwargs["sampler"].start, 60000 * 16 * 4 * 2)

    def test_dataloader_rejects_episode_groups_that_cross_batch_boundaries(self):
        config = SimpleNamespace(
            main=SimpleNamespace(
                cpu_workers_num=0,
                batch_size=8,
                seed=42,
            )
        )
        dataset = _TinyDataset()
        dataset.windows_per_episode = 3

        with self.assertRaisesRegex(
            ValueError, "batch_size=8.*windows_per_episode=3"
        ):
            build_dataloader(config, train_dataset=dataset)

    def test_only_pretrain_config_enables_grouped_episode_sampling(self):
        project_root = Path(__file__).resolve().parents[1]
        script_root = project_root / "scripts"
        pretrain = OmegaConf.load(
            script_root / "Pre_Train/pt_419h_gbs1024_100w_controlnet4_detach_true_mse.yaml"
        )
        arena_only = OmegaConf.load(
            script_root / "HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.yaml"
        )
        arena_ft = OmegaConf.load(
            script_root / "HumanoidArena_Multi_Task/ft_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.yaml"
        )

        self.assertEqual(pretrain.main.sampling.windows_per_episode, 8)
        self.assertEqual(arena_only.main.sampling.get("windows_per_episode", 1), 1)
        self.assertEqual(arena_ft.main.sampling.get("windows_per_episode", 1), 1)
        self.assertEqual(pretrain.main.batch_size % pretrain.main.sampling.windows_per_episode, 0)

    def test_training_launchers_do_not_hardcode_a_local_conda_environment(self):
        project_root = Path(__file__).resolve().parents[1]
        for name in (
            "Pre_Train/pt_419h_gbs1024_100w_controlnet4_detach_true_mse.sh",
            "HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.sh",
            "HumanoidArena_Multi_Task/ft_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.sh",
        ):
            script = (project_root / "scripts" / name).read_text(encoding="utf-8")
            self.assertNotIn("/home/CONNECT/", script)
            self.assertNotIn("/mnt/workspace/", script)
            self.assertIn("command -v accelerate", script)
            self.assertIn("KIMODO_ENV", script)
        pretrain_script = (
            project_root
            / "scripts/Pre_Train/pt_419h_gbs1024_100w_controlnet4_detach_true_mse.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('NUM_PROCESSES="${#GPU_IDS[@]}"', pretrain_script)
        self.assertIn('--num_processes "${NUM_PROCESSES}"', pretrain_script)

    def test_dataset_receives_opt_in_randomization_and_arena_defaults_to_off(self):
        project_root = Path(__file__).resolve().parents[1]
        for filename, enabled in (
            ("scripts/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.yaml", False),
            ("scripts/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand.yaml", True),
            ("scripts/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse.yaml", True),
        ):
            config = OmegaConf.load(project_root / filename)
            with self.subTest(filename=filename), patch("train.MultiSourceG1Dataset") as dataset:
                build_dataset(config)
                options = dataset.call_args.kwargs
                self.assertTrue(options["training"])
                if enabled:
                    self.assertEqual(options["domain_randomization"], OmegaConf.to_container(
                        config.main.domain_randomization, resolve=True
                    ))
                else:
                    self.assertIsNone(options["domain_randomization"])

    def test_resume_accepts_disabled_randomization_for_old_checkpoints(self):
        project_root = Path(__file__).resolve().parents[1]
        config = OmegaConf.load(project_root / "train.yaml")
        checkpoint_config = OmegaConf.to_container(config, resolve=True)
        checkpoint_config["main"].pop("domain_randomization", None)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps(checkpoint_config))
            for disabled in (None, {}, {"enabled": False}):
                with self.subTest(disabled=disabled):
                    config.main.domain_randomization = disabled
                    _validate_resume_config(config, directory)

    def test_resume_rejects_changed_randomization_but_finetuning_allows_it(self):
        project_root = Path(__file__).resolve().parents[1]
        config = OmegaConf.load(project_root / "train.yaml")
        config.main.domain_randomization = {
            "enabled": True,
            "color_jitter": {"probability": 0.8, "brightness": 0.2, "contrast": 0.2,
                             "saturation": 0.2, "hue": 0.02},
        }
        checkpoint_config = OmegaConf.to_container(config, resolve=True)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps(checkpoint_config))
            _validate_resume_config(config, directory)
            config.main.domain_randomization.color_jitter.brightness = 0.1
            with self.assertRaisesRegex(ValueError, "main.domain_randomization"):
                _validate_resume_config(config, directory)
            _validate_init_checkpoint_config(config, directory)
            config.main.domain_randomization.color_jitter.brightness = 0.2
            config.main.domain_randomization.view_crop = {
                "enabled": True, "probability": 0.3, "min_scale": 0.95,
            }
            with self.assertRaisesRegex(ValueError, "main.domain_randomization"):
                _validate_resume_config(config, directory)
            _validate_init_checkpoint_config(config, directory)
            config.main.domain_randomization.enabled = False
            with self.assertRaisesRegex(ValueError, "main.domain_randomization"):
                _validate_resume_config(config, directory)

    def test_resume_rejects_a_changed_episode_group_size(self):
        project_root = Path(__file__).resolve().parents[1]
        config = OmegaConf.load(project_root / "train.yaml")
        checkpoint_config = OmegaConf.to_container(config, resolve=True)
        checkpoint_config["main"]["sampling"]["windows_per_episode"] = 4

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory)
            (checkpoint_path / "config.json").write_text(
                json.dumps(checkpoint_config), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "main.sampling"):
                _validate_resume_config(config, str(checkpoint_path))

    def test_resume_treats_missing_motion_loss_fields_as_legacy_mse(self):
        project_root = Path(__file__).resolve().parents[1]
        config = OmegaConf.load(project_root / "train.yaml")
        checkpoint_config = OmegaConf.to_container(config, resolve=True)
        checkpoint_config["model"].pop("detach_root_control_for_body")
        checkpoint_loss = checkpoint_config["training"]["loss"]
        checkpoint_loss.pop("motion_loss_type")
        checkpoint_loss.pop("kimodo_smooth_l1_weights")
        checkpoint_mse_weights = checkpoint_loss.pop("mse_weights")
        checkpoint_loss.update(
            {
                "root_weight": checkpoint_mse_weights["root"],
                "body_weight": checkpoint_mse_weights["body"],
                "hand_weight": checkpoint_mse_weights["hand"],
                "hand_transition_weight": checkpoint_mse_weights[
                    "hand_transition"
                ],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory)
            (checkpoint_path / "config.json").write_text(
                json.dumps(checkpoint_config), encoding="utf-8"
            )

            _validate_resume_config(config, str(checkpoint_path))
            config.model.detach_root_control_for_body = True
            with self.assertRaisesRegex(
                ValueError, "model.detach_root_control_for_body"
            ):
                _validate_resume_config(config, str(checkpoint_path))
            config.model.detach_root_control_for_body = False
            config.training.loss.motion_loss_type = "kimodo_smooth_l1"
            with self.assertRaisesRegex(ValueError, "training.loss"):
                _validate_resume_config(config, str(checkpoint_path))

    def test_spawn_dataloader_can_fetch_a_batch(self):
        config = SimpleNamespace(
            main=SimpleNamespace(cpu_workers_num=2, batch_size=4, seed=42)
        )
        data_loader = build_dataloader(config, train_dataset=_TinyDataset())

        batch = next(iter(data_loader))

        self.assertEqual(batch.shape, (4,))


if __name__ == "__main__":
    unittest.main()
