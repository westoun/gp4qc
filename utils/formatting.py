#!/usr/bin/env python3

import numpy as np
from typing import List


def state_to_distribution(target_state: List[int]) -> List[float]:
    vectors = []
    for qubit_state in target_state:
        if qubit_state == 0:
            vectors.append([1, 0])
        else:
            vectors.append([0, 1])

    distribution = vectors[0]
    for vector in vectors[1:]:
        distribution = np.kron(distribution, vector)

    return distribution.tolist()
