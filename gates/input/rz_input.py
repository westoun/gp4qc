#!/usr/bin/env python3

from qiskit import QuantumCircuit
from typing import List

from .input import InputEncoding, InputEncodingConstructor


class RZEncoding(InputEncoding):
    name: str = "rz_input"

    _circuits: List[QuantumCircuit] = None
    _targets: List[int] = None

    def build_circuits(self, qubit_num: int, input_values: List[List[float]]) -> None:
        circuits: List[QuantumCircuit] = []

        for case_index in range(len(input_values)):
            circuit = QuantumCircuit(qubit_num)

            for i, qubit_value in enumerate(input_values[case_index]):
                circuit.rz(qubit_value, i)

            circuits.append(circuit)

        return circuits

    def __repr__(self) -> str:
        return f"{self.name}({','.join(['target' + str((i + 1)) + '=' + str(target) for i, target in enumerate(self._targets)])})"
