#!/usr/bin/env python3

from qiskit import QuantumCircuit
from random import randint, sample

from .gate import Gate


class Swap(Gate):
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

    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit:
        circuit.swap(self.target1, self.target2)
        return circuit
