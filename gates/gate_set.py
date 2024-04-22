#!/usr/bin/env python3

from random import choice
from typing import Type, List

from .gate import Gate
from .oracle import Oracle
from .utils import construct_gate_type_name


class GateSet:
    gates: List[Type[Gate]] = []
    gate_names: List[str] = []

    def __init__(self, gates: List[Type[Gate]], qubit_num: int) -> None:
        self.gates = gates
        self.gate_names = [construct_gate_type_name(gate) for gate in gates]
        self._qubit_num = qubit_num

    def random_gate(self) -> Gate:
        """Selects a gate type at random and initializes it."""

        GateType = choice(self.gates)
        gate = GateType(qubit_num=self._qubit_num)

        return gate

    def contains(self, gate: Type[Gate]) -> bool:
        gate_name = construct_gate_type_name(gate)
        return gate_name in self.gate_names

    def append(self, gate: Type[Gate]) -> None:
        if self.contains(gate):
            return

        self.gates.append(gate)
        self.gate_names.append(construct_gate_type_name(gate))

    def __repr__(self) -> str:
        representation = f"[{','.join([str(gate) for gate in self.gates])}]"
        return representation
