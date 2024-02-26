#!/usr/bin/env python3

from gates import Gate, Oracle, InputEncoding, HadamardLayer, Hadamard, OptimizableGate
from typing import List


def has_exactly_1_oracle(chromosome: List[Gate]) -> bool:
    oracle_gates = [gate for gate in chromosome if gate.is_oracle]
    return len(oracle_gates) == 1


def uses_oracle(chromosome: List[Gate]) -> bool:
    for gate in chromosome:
        if gate.is_oracle:
            return True

    return False


def has_exactly_1_input(chromosome: List[Gate]) -> bool:
    input_gates = [
        gate for gate in chromosome if gate.is_input
    ]
    return len(input_gates) == 1


def has_input_at_first_position(chromosome: List[Gate]) -> bool:
    return chromosome[0].is_input


def uses_hadamard(chromosome: List[Gate]) -> bool:
    for gate in chromosome:
        if type(gate) in [Hadamard, HadamardLayer]:
            return True

    return False


def uses_hadamard_layer(chromosome: List[Gate]) -> bool:
    for gate in chromosome:
        if type(gate) == HadamardLayer:
            return True

    return False


def uses_parametrized_gates(chromosome: List[Gate]) -> bool:
    parametrized_gates = [
        gate for gate in chromosome if gate.is_optimizable
    ]
    return len(parametrized_gates) > 0
