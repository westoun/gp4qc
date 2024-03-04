#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Tuple

from fitness import Fitness
from gates import Gate
from .params import OptimizerParams, default_params
from .optimizer import Optimizer
from .utils import (
    build_circuit,
    run_circuit,
    aggregate_state_distribution,
    get_state_distributions,
)


class DoNothingOptimizer(Optimizer):
    def __init__(
        self,
        target_distributions: List[List[float]],
        params: OptimizerParams = default_params,
    ) -> None:
        self.target_distributions = target_distributions
        self.params = params

    def optimize(
        self, chromosome: List[Gate], fitness: Fitness
    ) -> Tuple[List[Gate], float]:
        state_distributions: List[List[float]] = get_state_distributions(
            chromosome, params=self.params, case_count=len(self.target_distributions)
        )

        fitness_score = fitness.evaluate(
            state_distributions, self.target_distributions, chromosome
        )

        return chromosome, fitness_score
