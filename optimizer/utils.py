#!/usr/bin/env python3

from quasim import Circuit, QuaSim
from typing import List, Union, Tuple

from fitness import Fitness
from gates import Gate, MultiCaseGate, InputEncoding, Oracle, OptimizableGate
from .params import OptimizerParams


def run_circuit(circuit: Circuit) -> List[float]:
    simulator = QuaSim()
    simulator.evaluate_circuit(circuit)
    return circuit.probabilities


def build_circuit(
    chromosome: List[Gate],
    qubit_num: int,
    case_index=0,
) -> Circuit:
    circuit = Circuit(qubit_num)

    for gate in chromosome:
        if gate.is_multicase:
            gate.set_case_index(case_index)

        circuit = gate.apply_to(circuit)

    return circuit


def aggregate_state_distribution(
    state_distribution: List[float], measurement_qubit_num: int, ancillary_num: int
) -> List[float]:
    # This function assumes that ancillary qubits were added
    # after all "normal" qubits have been added.

    aggregated_distribution: List[float] = []
    for _ in range(2**measurement_qubit_num):
        aggregated_distribution.append(0.0)

    for i in range(2**measurement_qubit_num):
        for j in range(2**ancillary_num):
            aggregated_distribution[i] += state_distribution[i * 2**ancillary_num + j]

    return aggregated_distribution


def update_params(param_vector: List[float], chromosome: List[Gate]) -> List[Gate]:
    parametrized_gates = get_parametrized_gates(chromosome)

    for gate in parametrized_gates:
        gate_params, param_vector = (
            param_vector[: gate.param_count],
            param_vector[gate.param_count :],
        )
        gate.set_params(gate_params)

    return chromosome


def get_parametrized_gates(chromosome: List[Gate]) -> List[OptimizableGate]:
    parametrized_gates = [gate for gate in chromosome if gate.is_optimizable]
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
