#!/usr/bin/env python3

from functools import partial
from scipy.optimize import minimize, OptimizeResult
from typing import List, Tuple, Union

from fitness import Fitness
from gates import Gate, OptimizableGate
from .params import OptimizerParams, default_params
from .optimizer import Optimizer
from .utils import build_circuit, run_circuit, aggregate_state_distribution


# TODO: Move to utils
def update_params(param_vector: List[float], chromosome: List[Gate]) -> List[Gate]:
    parametrized_gates = get_parametrized_gates(chromosome)

    for gate in parametrized_gates:
        gate_params, param_vector = (
            param_vector[: gate.param_count],
            param_vector[gate.param_count :],
        )
        gate.set_params(gate_params)

    return chromosome


# TODO: add caching
def get_parametrized_gates(chromosome: List[Gate]) -> List[OptimizableGate]:
    parametrized_gates = [
        gate for gate in chromosome if issubclass(gate.__class__, OptimizableGate)
    ]
    return parametrized_gates


def has_parametrized_gates(chromosome: List[Gate]) -> bool:
    parametrized_gates = get_parametrized_gates(chromosome)
    return len(parametrized_gates) > 0


def extract_param_vector(chromosome: List[Gate]) -> List[float]:
    parametrized_gates = get_parametrized_gates(chromosome)

    param_vector = []
    for gate in parametrized_gates:
        param_vector.extend(gate.params)

    return param_vector


def extract_bounds(chromosome: List[Gate]) -> List[Union[Tuple[float, float], None]]:
    parametrized_gates = get_parametrized_gates(chromosome)

    bounds = []
    for gate in parametrized_gates:
        bounds.extend(gate.bounds)

    return bounds


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


def get_state_distributions(
    chromosome: List[Gate], params: OptimizerParams, case_count: int = 1
):
    state_distributions: List[List[float]] = []

    for i in range(case_count):
        circuit = build_circuit(
            chromosome,
            qubit_num=params.qubit_num,
            case_index=i,
        )

        state_distribution = run_circuit(circuit)

        state_distribution = aggregate_state_distribution(
            state_distribution,
            measurement_qubit_num=params.measurement_qubit_num,
            ancillary_num=params.qubit_num - params.measurement_qubit_num,
        )

        state_distributions.append(state_distribution)

    return state_distributions


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
