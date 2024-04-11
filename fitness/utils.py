#!/usr/bin/env python3

from typing import List

from gates import Gate


def get_gate_count(chromosome: List[Gate]) -> int:
    return sum([gate.gate_count for gate in chromosome])
