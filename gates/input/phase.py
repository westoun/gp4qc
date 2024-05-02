#!/usr/bin/env python3

from quasim import Circuit
from quasim.gates import Phase
from typing import List

from .input import InputEncoding, InputEncodingConstructor


class PhaseEncoding(InputEncoding):
    name: str = "phase_input"

    _circuits: List[Circuit] = None
    _targets: List[int] = None

    def build_circuits(self, qubit_num: int, input_values: List[List[float]]) -> None:
        circuits: List[Circuit] = []

        for case_index in range(len(input_values)):
            circuit = Circuit(qubit_num)

            for i, qubit_value in enumerate(input_values[case_index]):
                circuit.apply(Phase(i, theta=qubit_value))

            circuits.append(circuit)

        return circuits

    def __repr__(self) -> str:
        return f"{self.name}({','.join(['target' + str((i + 1)) + '=' + str(target) for i, target in enumerate(self._targets)])})"
