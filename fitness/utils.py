#!/usr/bin/env python3

from typing import List

from gates import Gate


def count_gates(chromosome: List[Gate]) -> int:
    return sum([gate.gate_count for gate in chromosome])
