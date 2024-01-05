#!/usr/bin/env python3

from qiskit import QuantumCircuit
from random import randint, sample

from .gate import Gate


class CCX(Gate):
    controll1: int
    controll2: int
    target: int

    def __init__(self, qubit_num: int):
        assert (
            qubit_num > 2
        ), "The CC X Gate requires at least 2 qubits to operate as intended."

        self._qubit_num = qubit_num
        self.controll1, self.controll2, self.target = sample(range(0, qubit_num), 3)

    def mutate_operands(self) -> None:
        self.controll1, self.controll2, self.target = sample(
            range(0, self._qubit_num), 3
        )

    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit:
        circuit.ccx(self.controll1, self.controll2, self.target)
        return circuit

    def __repr__(self) -> str:
        return f"ccx(control1={self.controll1},control2={self.controll2},target={self.target})"
