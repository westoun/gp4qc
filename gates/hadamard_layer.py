#!/usr/bin/env python3

from qiskit import QuantumCircuit

from .gate import Gate


class HadamardLayer(Gate):

    def __init__(self, qubit_num: int):
        self._qubit_num = qubit_num

    def mutate_operands(self) -> None:
        pass 

    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit:
        for i in range(self._qubit_num):
            circuit.h(i)
        return circuit

    def __repr__(self) -> str:
        return f"h_layer()"
