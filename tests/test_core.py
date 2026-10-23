import hashlib, json, sys, unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pa_tsmixer.data import split_code
from pa_tsmixer.models import PATSMixer, TSMixer
from kinematic_baselines import predict

count = lambda m: sum(p.numel() for p in m.parameters())


class Models(unittest.TestCase):
    def test_parameter_counts_match_the_paper(self):
        self.assertEqual(count(PATSMixer(30)), 44_672)
        self.assertEqual(count(TSMixer(30)), 34_724)
        self.assertEqual(count(TSMixer(30, d_model=167)), 44_708)
        self.assertEqual(count(PATSMixer(30, uniform_width=96)), 44_800)

    def test_output_shape(self):
        for h in (10, 20, 30, 60):
            self.assertEqual(PATSMixer(h)(torch.zeros(2, 60, 3)).shape, (2, h, 3))

    def test_residual_output_adds_the_last_observation(self):
        torch.manual_seed(0)
        residual, direct = PATSMixer(30), PATSMixer(30, last_residual=False)
        direct.load_state_dict(residual.state_dict())
        x = torch.randn(2, 60, 3)
        with torch.no_grad():
            self.assertTrue(torch.allclose(residual.eval()(x), direct.eval()(x) + x[:, -1:]))
        self.assertFalse(TSMixer(30).last_residual)

    def test_equal_weight_control_freezes_the_scale_logits(self):
        m = PATSMixer(30, learn_scale_weights=False)
        self.assertTrue(all(not b.temporal.scale_w.requires_grad for b in m.model))

    def test_released_checkpoints(self):
        index = json.loads((ROOT / "checkpoints/index.json").read_text(encoding="utf-8"))
        self.assertEqual(len(index), 108)
        for e in index:
            self.assertEqual(hashlib.sha256((ROOT / "checkpoints" / e["file"]).read_bytes()).hexdigest(), e["sha256"])


class Protocol(unittest.TestCase):
    def test_split_rule_is_fixed(self):
        codes = [split_code(i) for i in range(10_000)]
        self.assertEqual([split_code(i) for i in (0, 1, 2, 3)], codes[:4])
        shares = np.bincount(codes, minlength=3) / len(codes)
        self.assertTrue(np.allclose(shares, [.70, .15, .15], atol=.02))

    def test_motion_groups_on_synthetic_histories(self):
        from pa_tsmixer.motion import groups
        t = np.arange(60, dtype=np.float64)
        straight_climb = np.column_stack((30 * t, 0 * t, 3 * t))
        theta = np.radians(2.) * t                                  # 2 deg/s turn at 30 m/s
        turn = np.column_stack((np.cumsum(30 * np.cos(theta)), np.cumsum(30 * np.sin(theta)), 0 * t))
        slow = np.column_stack((t, 0 * t, -2 * t))
        g = groups(np.stack([straight_climb, turn, slow]))
        self.assertEqual(g["climb"].tolist(), [True, False, False])
        self.assertEqual(g["descent"].tolist(), [False, False, True])
        self.assertEqual(g["turning"].tolist(), [False, True, False])
        self.assertEqual(g["nonturning"].tolist(), [True, False, False])
        self.assertEqual(g["low_speed"].tolist(), [False, False, True])

    def test_kinematic_baselines_are_exact_on_their_own_models(self):
        t = np.arange(-59, 61, dtype=np.float64)[:, None]
        line = (np.array([[3., -2., .5]]) * t)[None]
        accel = (np.array([[1., 0., -1.]]) * t + np.array([[.2, .1, 0.]]) * .5 * t * t)[None]
        self.assertTrue(np.allclose(predict(line[:, :60], 60, "CV", 5), line[:, 60:]))
        self.assertTrue(np.allclose(predict(accel[:, :60], 60, "CA", 20), accel[:, 60:]))


if __name__ == "__main__":
    unittest.main()
