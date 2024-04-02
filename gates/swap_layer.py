#!/usr/bin/env python3

from qiskit import QuantumCircuit

from .gate import Gate


class SwapLayer(Gate):
    name: str = "swap_layer"

    def __init__(self, qubit_num: int):
        self._qubit_num = qubit_num

    def mutate_operands(self) -> None:
        pass

    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit:
        for i in range(self._qubit_num // 2):
            target1 = i
            target2 = self._qubit_num - 1 - i
            circuit.swap(target1, target2)
        return circuit

    def __repr__(self) -> str:
        return f"{self.name}()"
