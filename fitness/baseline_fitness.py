#!/usr/bin/env python3

from scipy.spatial import distance
from statistics import mean
from typing import List, Tuple

from gates import Gate, InputEncoding, Identity
from .fitness import Fitness
from .params import FitnessParams, default_params
from .utils import count_gates, count_gate_types


class BaselineFitness(Fitness):
    def __init__(self, params: FitnessParams = default_params) -> None:
        self.params = params

    def evaluate(
        self,
        state_distributions: List[List[float]],
        target_distributions: List[List[float]],
        chromosome: List[Gate],
    ) -> float:
        hits: int = len(target_distributions)
        errors: List[float] = []

        for i, (state_distribution, target_distribution) in enumerate(
            zip(state_distributions, target_distributions)
        ):
            assert len(state_distribution) == len(
                target_distribution
            ), f"Missmatch between produced distribution (len {len(state_distribution)}) and target distribution (len {len(target_distribution)})"

            match_index = target_distribution.index(1.0)
            assert (
                match_index != -1
            ), f"Check the formatting of your target distributions. A 1 is missing in the {i + 1}. target distribution."

            probability = state_distribution[match_index]
            if probability >= 2 / 3:  # Use 2/3 since threshold for BPP complexity class
                hits -= 1
            else:
                error = distance.jensenshannon(state_distribution, target_distribution)
                errors.append(error)

        fitness_score: float = 0
        if len(errors) > 0:
            fitness_score = hits + sum(errors) / max(
                hits, 1
            )  # ranges from (1, #testcases]
        else:
            fitness_score = count_gates(chromosome) / 100000

        for validity_check in self.params.validity_checks:
            if not validity_check(chromosome):
                fitness_score += 100
                break

        return fitness_score
