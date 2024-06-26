#!/usr/bin/env python3

from typing import List, Type

from gates import Gate, Identity, CombinedGate


def count_gates(chromosome: List[Gate]) -> int:
    return sum([gate.gate_count for gate in chromosome])


def count_gate_types(chromosome: List[Gate]) -> int:
    gate_names = []

    for gate in chromosome:
        if type(gate) == Identity:
            continue
        elif type(gate) == CombinedGate:
            name = "_".join([GateType.name for GateType in gate.GateTypes])
            gate_names.append(name)
        else:
            gate_names.append(gate.name)

    return len(set(gate_names))


def contains_gate_type(chromosome: List[Gate], GateTypes: List[Type]) -> bool:
    for gate in chromosome:
        if type(gate) in GateTypes:
            return True

        if type(gate) == CombinedGate:
            for GateType in gate.GateTypes:
                if GateType in GateTypes:
                    return True

    return False


def count_gate_calls(chromosome: List[Gate], GateType: Type) -> int:
    count = 0

    for gate in chromosome:
        if type(gate) == GateType:
            count += 1

        elif type(gate) == CombinedGate:
            for GateType_ in gate.GateTypes:
                if GateType_ == GateType:
                    count += 1

    return count
