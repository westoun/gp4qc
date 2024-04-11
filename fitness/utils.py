#!/usr/bin/env python3

from typing import List

from gates import Gate, Identity


def count_gates(chromosome: List[Gate]) -> int:
    return sum([gate.gate_count for gate in chromosome])


def count_gate_types(chromosome: List[Gate]) -> int:
    gate_names = [gate.name for gate in chromosome]

    return len(set(gate_names))
