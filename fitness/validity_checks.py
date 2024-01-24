#!/usr/bin/env python3

from gates import Gate, Oracle, InputEncoding, HadamardLayer, Hadamard
from typing import List


def has_exactly_1_oracle(chromosome: List[Gate]) -> bool:
    oracle_gates = [gate for gate in chromosome if issubclass(gate.__class__, Oracle)]
    return len(oracle_gates) == 1


def has_exactly_1_input(chromosome: List[Gate]) -> bool:
    input_gates = [
        gate for gate in chromosome if issubclass(gate.__class__, InputEncoding)
    ]
    return len(input_gates) == 1


def has_input_at_first_position(chromosome: List[Gate]) -> bool:
    return issubclass(chromosome[0].__class__, InputEncoding)

def uses_hadamard(chromosome: List[Gate]) -> bool:
    for gate in chromosome:
        if type(gate) in [Hadamard, HadamardLayer]:
            return True 

    return False 