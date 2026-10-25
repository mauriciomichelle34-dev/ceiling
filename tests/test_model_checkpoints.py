"""Load shipped checkpoints through the production predictor, without devices."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_app import DEFAULT_MODEL, Predictor
from inference import predict_probability


class ShippedCheckpointTests(unittest.TestCase):
    def check_checkpoint(self, path, expected_hash):
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash)
        with tempfile.TemporaryDirectory() as folder:
            predictor = Predictor(path, Path(folder))
            source = Image.fromarray(np.full((192,288,3),128,np.uint8))
            probability = predict_probability(predictor.network, source, predictor.config, predictor.device)
            self.assertEqual(probability.shape, (192,288))
            self.assertTrue(np.isfinite(probability).all())
            self.assertTrue(((probability >= 0) & (probability <= 1)).all())
            self.assertEqual(predictor.config['preprocessing'], 'letterbox_rgb_0_1')
            self.assertEqual(predictor.config['threshold'], .5)

    def test_default_v4_loads_and_predicts_at_source_size(self):
        self.assertEqual(DEFAULT_MODEL.name, 'mycelium_v4_full16.pt')
        self.check_checkpoint(DEFAULT_MODEL, '469ccfc32925078fcbda98239d385803c7361a189ad455b7b2653de75062ac45')

    def test_v3_rollback_remains_loadable_and_unchanged(self):
        self.check_checkpoint(ROOT/'mycelium_v3.pt', 'b62dd4b9595ea27d9e48a19543ea6a2b1d7341d86b38026f12c822023b8fe180')


if __name__ == '__main__':
    unittest.main(verbosity=2)
