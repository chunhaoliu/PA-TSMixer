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

    def test_equal_weight_control_freezes_the_scale_logits(self):
        m = PATSMixer(30, learn_scale_weights=False)
        self.assertTrue(all(not b.temporal.scale_w.requires_grad for b in m.model))

    def test_released_checkpoints(self):
        index = json.loads((ROOT / "checkpoints/index.json").read_text(encoding="utf-8"))
        self.assertEqual(len(index), 96)
        for e in index:
            self.assertEqual(hashlib.sha256((ROOT / "checkpoints" / e["file"]).read_bytes()).hexdigest(), e["sha256"])


class Protocol(unittest.TestCase):
    def test_split_rule_is_fixed(self):
        codes = [split_code(i) for i in range(10_000)]
        self.assertEqual([split_code(i) for i in (0, 1, 2, 3)], codes[:4])
        shares = np.bincount(codes, minlength=3) / len(codes)
        self.assertTrue(np.allclose(shares, [.70, .15, .15], atol=.02))

    def test_kinematic_baselines_are_exact_on_their_own_models(self):
        t = np.arange(-59, 61, dtype=np.float64)[:, None]
        line = (np.array([[3., -2., .5]]) * t)[None]
        accel = (np.array([[1., 0., -1.]]) * t + np.array([[.2, .1, 0.]]) * .5 * t * t)[None]
        self.assertTrue(np.allclose(predict(line[:, :60], 60, "CV", 5), line[:, 60:]))
        self.assertTrue(np.allclose(predict(accel[:, :60], 60, "CA", 20), accel[:, 60:]))


if __name__ == "__main__":
    unittest.main()
