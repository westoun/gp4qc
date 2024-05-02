#!/usr/bin/env python3

from quasim import Circuit
from quasim.gates import Y as YGate
from random import randint

from .gate import Gate


class Y(Gate):
    name: str = "y"

    target: int

    def __init__(self, qubit_num: int):
        self._qubit_num = qubit_num
        self.target = randint(0, qubit_num - 1)

    def mutate_operands(self) -> None:
        self.target = randint(0, self._qubit_num - 1)

    def apply_to(self, circuit: Circuit) -> Circuit:
        circuit.apply(YGate(self.target))
        return circuit

    def __repr__(self) -> str:
        return f"{self.name}(target={self.target})"
