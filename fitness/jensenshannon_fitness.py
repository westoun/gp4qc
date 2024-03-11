#!/usr/bin/env python3

from scipy.spatial import distance
from statistics import mean
from typing import List, Tuple

from gates import Gate, InputEncoding
from .fitness import Fitness
from .params import FitnessParams, default_params


class Jensensshannon(Fitness):
    def __init__(self, params: FitnessParams = default_params) -> None:
        self.params = params

    def evaluate(
        self,
        state_distributions: List[List[float]],
        target_distributions: List[List[float]],
        chromosome: List[Gate],
    ) -> float:
        errors: List[float] = []

        for state_distribution, target_distribution in zip(
            state_distributions, target_distributions
        ):
            assert len(state_distribution) == len(
                target_distribution
            ), f"Missmatch between produced distribution (len {len(state_distribution)}) and target distribution (len {len(target_distribution)})"

            error = distance.jensenshannon(state_distribution, target_distribution)
            errors.append(error)

        error = mean(errors)

        for validity_check in self.params.validity_checks:
            if not validity_check(chromosome):
                error += 100
                break

        return error
