"""Exercise the FT launcher without starting Accelerate or allocating a GPU."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'scripts/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse.sh'


class RealWorldLauncherTest(unittest.TestCase):
    def launch(self, resume):
        with tempfile.TemporaryDirectory(prefix='kimodo launcher ') as directory:
            root = Path(directory)
            binary = root / 'env/bin/accelerate'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/usr/bin/env bash\nprintf "GPU=%s\\n" "$CUDA_VISIBLE_DEVICES"\nprintf "%s\\n" "$@"\n')
            binary.chmod(0o755)
            checkpoint = root / 'checkpoint_20000'
            checkpoint.mkdir()
            for name in ('config.json', 'training_state.pt'):
                (checkpoint / name).touch()
            env = {k: v for k, v in os.environ.items() if not k.startswith('KIMODO_')}
            env.update(KIMODO_ENV=str(root / 'env'), KIMODO_GPUS='0,1,2,3,4,5,6,7', KIMODO_RESUME=resume)
            result = subprocess.run(['bash', str(LAUNCHER), str(checkpoint), 'teleop-boxing'], env=env, capture_output=True, text=True, check=True)
            lines = result.stdout.splitlines()
            self.assertIn('GPU=0,1,2,3,4,5,6,7', lines)
            self.assertEqual(lines[lines.index('--num_processes') + 1], '8')
            return lines

    def test_new_finetune_initializes_weights(self):
        lines = self.launch('0')
        self.assertIn('--init-checkpoint', lines)
        self.assertNotIn('--resume', lines)

    def test_resume_restores_training_state(self):
        lines = self.launch('1')
        self.assertIn('--resume', lines)
        self.assertNotIn('--init-checkpoint', lines)


if __name__ == '__main__':
    unittest.main()
