#!/usr/bin/env python3

from functools import partial
from scipy.optimize import minimize, OptimizeResult
from typing import List, Tuple, Union

from fitness import Fitness
from gates import Gate, OptimizableGate
from .params import OptimizerParams, default_params
from .optimizer import Optimizer
from .utils import (
    has_parametrized_gates,
    get_state_distributions,
    extract_bounds,
    extract_param_vector,
    update_params,
)


def evaluate(
    param_vector: List[float],
    chromosome: List[Gate],
    fitness: Fitness,
    target_distributions: List[List[float]],
    params: OptimizerParams,
) -> float:
    chromosome = update_params(param_vector, chromosome)

    state_distributions = get_state_distributions(
        chromosome, params=params, case_count=len(target_distributions)
    )

    fitness_score = fitness.evaluate(
        state_distributions, target_distributions, chromosome
    )
    return fitness_score


class NumericalOptimizer(Optimizer):
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
        if not has_parametrized_gates(chromosome):
            state_distributions = get_state_distributions(
                chromosome,
                params=self.params,
                case_count=len(self.target_distributions),
            )

            fitness_score = fitness.evaluate(
                state_distributions, self.target_distributions, chromosome
            )

            return chromosome, fitness_score

        initial_params = extract_param_vector(chromosome)
        bounds = extract_bounds(chromosome)

        objective_function = partial(
            evaluate,
            chromosome=chromosome,
            fitness=fitness,
            target_distributions=self.target_distributions,
            params=self.params,
        )

        optimization_result: OptimizeResult = minimize(
            objective_function,
            x0=initial_params,
            method="Nelder-Mead",
            bounds=bounds,
            tol=self.params.tolerance,
            options={"maxiter": self.params.max_iter, "disp": False},
        )

        best_params = optimization_result.x
        chromosome = update_params(best_params, chromosome)

        fitness_score = optimization_result.fun.tolist()

        return chromosome, fitness_score
