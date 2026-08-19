from __future__ import annotations

import unittest

import numpy as np

from learning2hear.egograph import EgoGraphConfig, run_egograph, self_anchor


class EgoGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(7)
        positive = generator.normal(0.0, 0.08, size=(50, 12)).astype(np.float32)
        negative = generator.normal(0.0, 0.08, size=(50, 12)).astype(np.float32)
        positive[:, 0] += 1.0
        negative[:, 0] -= 0.35
        negative[:, 1] += 1.0
        self.embeddings = np.concatenate([positive, negative], axis=0)

    def test_self_anchor_is_unit_normalized(self) -> None:
        anchor = self_anchor(self.embeddings)
        self.assertAlmostEqual(float(np.linalg.norm(anchor)), 1.0, places=5)

    def test_scores_match_paper_configuration(self) -> None:
        result = run_egograph(
            self.embeddings,
            EgoGraphConfig(pca_dim=8, n_neighbors=12, n_jobs=1),
        )
        self.assertEqual(result.scores.shape, (100,))
        self.assertTrue(np.all((0.0 <= result.scores) & (result.scores <= 1.0)))
        self.assertEqual(result.diagnostics["positive_seed_count"], 10)
        self.assertEqual(result.diagnostics["negative_seed_count"], 10)
        self.assertGreater(
            float(result.scores[:50].mean()),
            float(result.scores[50:].mean()),
        )


if __name__ == "__main__":
    unittest.main()
