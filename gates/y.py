#!/usr/bin/env python3

from qiskit import QuantumCircuit
from random import randint

from .gate import Gate


class Y(Gate):
    target: int

    def __init__(self, qubit_num: int):
        self._qubit_num = qubit_num
        self.target = randint(0, qubit_num - 1)

    def mutate_operands(self) -> None:
        self.target = randint(0, self._qubit_num - 1)

    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit:
        circuit.y(self.target)
        return circuit
