import unittest

import numpy as np

from src.correlation_analysis import benjamini_hochberg
from src.data_loader import parse_formula, reconstruct_formula_descriptors
from src.models import evaluate_predictions
from src.sensitivity_experiments import _subset_param_grid


class CoreAnalysisTests(unittest.TestCase):
    def test_formula_parser(self):
        self.assertEqual(parse_formula("Li2C2O4"), {"Li": 2, "C": 2, "O": 4})

    def test_reconstructed_framework_descriptors(self):
        descriptors = reconstruct_formula_descriptors("Li5FeO4")
        self.assertAlmostEqual(descriptors["reconstructed_framework_oxygen_fraction"], 0.8)
        self.assertAlmostEqual(descriptors["reconstructed_framework_electronegativity_std"], 0.644, places=3)

    def test_bh_is_monotonic_after_sorting(self):
        q_values = benjamini_hochberg(np.array([0.04, 0.001, 0.2]))
        self.assertTrue(np.all((q_values >= 0) & (q_values <= 1)))
        self.assertLessEqual(q_values[1], q_values[0])

    def test_metrics(self):
        metrics = evaluate_predictions(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
        self.assertAlmostEqual(metrics["MAE"], 0.5)
        self.assertAlmostEqual(metrics["MaxAE"], 1.0)

    def test_subset_grid_is_nested_for_log_target(self):
        grid = _subset_param_grid("Ridge", 5, 23, 5, "log", [1, 2, "all"])
        self.assertIn("regressor__selector__k", grid)
        self.assertIn("regressor__model__alpha", grid)
        self.assertEqual(grid["regressor__selector__k"], [1, 2, "all"])


if __name__ == "__main__":
    unittest.main()
