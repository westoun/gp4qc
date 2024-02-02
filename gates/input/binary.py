#!/usr/bin/env python3

from qiskit import QuantumCircuit
from typing import List

from .input import InputEncoding

class BinaryEncoding(InputEncoding):
    name: str = "x_input"

    @classmethod
    def init_circuits(
        cls, input_values: List[List[int]], qubit_num: int
    ) -> "BinaryEncoding":
        cls._targets = list(range(qubit_num))

        circuits: List[QuantumCircuit] = []

        for case_index in range(len(input_values)):
            circuit = QuantumCircuit(qubit_num)

            for i, qubit_value in enumerate(input_values[case_index]):
                if qubit_value == 0:
                    continue
                elif qubit_value == 1:
                    circuit.x(i)
                else:
                    raise NotImplementedError()

            circuits.append(circuit)

        cls._circuits = circuits
        return cls

    def __repr__(self) -> str:
        return f"{self.name}({','.join(['target' + str((i + 1)) + '=' + str(target) for i, target in enumerate(self._targets)])})"
