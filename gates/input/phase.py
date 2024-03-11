#!/usr/bin/env python3

from qiskit import QuantumCircuit
from typing import List

from .input import InputEncoding, InputEncodingConstructor


class PhaseEncoding(InputEncoding):
    name: str = "phase_input"

    _circuits: List[QuantumCircuit] = None
    _targets: List[int] = None

    def build_circuits(self, qubit_num: int, input_values: List[List[float]]) -> None:
        circuits: List[QuantumCircuit] = []

        for case_index in range(len(input_values)):
            circuit = QuantumCircuit(qubit_num)

            for i, qubit_value in enumerate(input_values[case_index]):
                circuit.p(qubit_value, i)

            circuits.append(circuit)

        return circuits

    def __repr__(self) -> str:
        return f"{self.name}({','.join(['target' + str((i + 1)) + '=' + str(target) for i, target in enumerate(self._targets)])})"


class PhaseEncodingConstructor(InputEncodingConstructor):
    input_values: List[List[float]] = None

    def __init__(self, input_values: List[List[float]]) -> None:
        self.input_values = input_values

    def __call__(self, qubit_num: int) -> PhaseEncoding:
        return PhaseEncoding(qubit_num, input_values=self.input_values)
