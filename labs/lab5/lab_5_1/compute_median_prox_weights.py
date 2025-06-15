from typing import Optional, Union, List, Dict

import numpy as np


def _compute_median_proximity_weights(
        feature_ranks: Union[List[float], Dict[str, float]]
) -> Optional[Union[List[float], Dict[str, float]]]:
    """
    Calculates and normalizes weights based on proximity to the median rank.
    Supports either a list of floats or a dict mapping keys to floats,
    returning the same type.

    Objective:
    - Compute weights reflecting proximity to the median rank.

    Steps:
    1. Extract values into a NumPy array and initialize alpha (small constant).
    2. If no features have non-zero ranks, return None.
    3. Compute the median of non-zero values.
    4. Compute weights via: weight = alpha / (alpha + |rank - median|).
    5. Normalize weights so they sum to 1.
    6. Map weights back to the original type.

    Parameters:
    - feature_ranks: List[float] or Dict[str, float]

    Returns:
    - List[float] or Dict[str, float] of normalized weights,
      or None if computation is not possible.
    """

    ### YOUR CODE HERE ###




    ### END YOUR CODE HERE ###