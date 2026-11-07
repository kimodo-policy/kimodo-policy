"""Run in the H20 kimodopolicy environment; no GPU or robot required."""
import unittest
import numpy as np
import torch
import real_world_server as server
from motion.g1_reference import HumanoidArenaActionDecoder, CANONICAL_G1_JOINT_NAMES_29
from motion.representation.kimodo_motionrep import KimodoMotionRep
from skeleton.definitions import G1Skeleton34

class PreprocessingEquivalence(unittest.TestCase):
    def test_zero_root_fast_path_matches_full_representation(self):
        torch.set_num_threads(1)
        decoder=HumanoidArenaActionDecoder(G1Skeleton34(),server.G1_XML_PATH,fps=50)
        rep=KimodoMotionRep(G1Skeleton34(),fps=30,stats_path=None)
        generator=torch.Generator().manual_seed(7)
        for count in (5,17,101,172):
            q=torch.randn(count,29,generator=generator)*.2
            root=torch.zeros(count,3)
            rotation=torch.eye(3).repeat(count,1,1)
            pose=decoder.decode_joint_configuration_pose(q,root,root_rotation_matrices=rotation,joint_names=CANONICAL_G1_JOINT_NAMES_29)
            complete=decoder.decode_joint_configuration(q,root,root_rotation_matrices=rotation,joint_names=CANONICAL_G1_JOINT_NAMES_29)
            for key in pose: torch.testing.assert_close(pose[key],complete[key],rtol=0,atol=0)
            for offset in (0., .2):
                xyz=pose['root_positions']+offset
                expected=rep(pose['local_rot_mats'],xyz,to_normalize=False).cpu()
                actual=server._kimodo_features_from_motion(pose['local_rot_mats'],xyz,rep)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)

if __name__=='__main__': unittest.main()
