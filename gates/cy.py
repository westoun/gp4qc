#!/usr/bin/env python3

from quasim import Circuit
from quasim.gates import CY as CYGate
from random import randint, sample

from .gate import Gate


class CY(Gate):
    name: str = "cy"

    controll: int
    target: int

    def __init__(self, qubit_num: int):
        assert (
            qubit_num > 1
        ), "The Controlled Y Gate requires at least 2 qubits to operate as intended."

        self._qubit_num = qubit_num
        self.target, self.controll = sample(range(0, qubit_num), 2)

    def mutate_operands(self) -> None:
        self.target, self.controll = sample(range(0, self._qubit_num), 2)

    def apply_to(self, circuit: Circuit) -> Circuit:
        circuit.apply(CYGate(self.controll, self.target))
        return circuit

    def __repr__(self) -> str:
        return f"{self.name}(control={self.controll},target={self.target})"
