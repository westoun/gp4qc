#!/usr/bin/env python3

from random import choice
from typing import Type, List

from .gate import Gate
from .oracle import Oracle


class GateSet:
    gates: List[Type[Gate]] = []

    def __init__(self, gates: List[Type[Gate]], qubit_num: int) -> None:
        self.gates = gates
        self._qubit_num = qubit_num

    def random_gate(self) -> Gate:
        """Selects a gate type at random and initializes it."""

        GateType = choice(self.gates)
        gate = GateType(qubit_num=self._qubit_num)

        return gate

    def append(self, gate: Type[Gate]) -> None:
        self.gates.append(gate)
