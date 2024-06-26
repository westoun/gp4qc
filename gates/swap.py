#!/usr/bin/env python3

from quasim import Circuit
from quasim.gates import Swap as SwapGate
from random import randint, sample

from .gate import Gate


class Swap(Gate):
    name: str = "swap"

    target1: int
    target2: int

    def __init__(self, qubit_num: int):
        assert (
            qubit_num > 1
        ), "The Swap Gate requires at least 2 qubits to operate as intended."

        self._qubit_num = qubit_num
        self.target1, self.target2 = sample(range(0, qubit_num), 2)

    def mutate_operands(self) -> None:
        self.target1, self.target2 = sample(range(0, self._qubit_num), 2)

    def apply_to(self, circuit: Circuit) -> Circuit:
        circuit.apply(SwapGate(self.target1, self.target2))
        return circuit

    def __repr__(self) -> str:
        return f"{self.name}(target1={self.target1},target2={self.target2})"
