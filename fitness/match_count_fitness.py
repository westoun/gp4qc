#!/usr/bin/env python3

from scipy.spatial import distance
from statistics import mean
from typing import List, Tuple

from gates import Gate, InputEncoding
from .fitness import Fitness
from .params import FitnessParams, default_params


class MatchCount(Fitness):
    def __init__(
        self, params: FitnessParams = default_params
    ) -> None:
        self.params = params

    def evaluate(self, state_distributions: List[List[float]], target_distributions: List[List[float]], 
                 chromosome: List[Gate]) -> float:
        match_count: int = 0

        for state_distribution, target_distribution in zip(state_distributions, target_distributions):
            assert len(state_distribution) == len(target_distribution)

            match_index = target_distribution.index(1.0)
            assert (
                match_index != -1
            ), f"Check the formatting of your target distributions. A 1 is missing in the {i + 1}. distribution."

            probability = state_distribution[match_index]
            if probability > 0.5:
                match_count += 1

        error = (len(target_distributions) - match_count) / len(
            target_distributions
        )

        for validity_check in self.params.validity_checks:
            if not validity_check(chromosome):
                error += 100
                break

        return error
