import unittest

from labs.lab5.lab_5_1.compute_median_prox_weights import _compute_median_proximity_weights


class TestComputeMedianProximityWeights(unittest.TestCase):

    def test_custom_case(self):
        """Test with a custom case where feature ranks are specified along with their expected weights."""
        # Specify the input feature ranks here
        feature_rank_dict = {
            'feature1': 0.15,
            'feature2': 0.75,
            'feature3': 0.25,
            'feature4': 0.60,
            'feature5': 0.05
        }

        # Manually specify the expected weights for each feature based on their proximity to the median
        expected_weights = {
            'feature1': 0.009709507681326553,  # Placeholder values, replace with your calculated expected weights
            'feature2': 0.0019574057401476686,
            'feature3': 0.980660275813982,
            'feature4': 0.0027939039196979544,
            'feature5': 0.004878906844845681
        }

        weights = _compute_median_proximity_weights(feature_rank_dict)
        self.assertIsNotNone(weights, "Weights should not be None")
        self.assertEqual(len(weights), len(feature_rank_dict), "Weights dictionary should have the same size as input")

        # Ensure the sum of weights is close to 1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5, msg="Weights should sum to 1")

        # Verify each computed weight against its expected value
        for key, expected_weight in expected_weights.items():
            self.assertAlmostEqual(weights[key], expected_weight, places=5,
                                   msg=f"Weight for {key} does not match expected value")

    def test_empty_input(self):
        """Test with an empty input dictionary."""
        feature_rank_dict = {}
        weights = _compute_median_proximity_weights(feature_rank_dict)
        self.assertIsNone(weights, "Weights should be None for empty input")

    def test_all_zero_ranks(self):
        """Test with all feature ranks set to zero."""
        feature_rank_dict = {'feature1': 0, 'feature2': 0, 'feature3': 0, 'feature4': 0}
        weights = _compute_median_proximity_weights(feature_rank_dict)
        self.assertIsNone(weights, "Weights should be None when all ranks are zero")


if __name__ == '__main__':
    unittest.main()
