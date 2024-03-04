#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractclassmethod
from qiskit import QuantumCircuit
from typing import Any, List

from gates.multicase_gate import MultiCaseGate


class InputEncoding(MultiCaseGate, ABC):
    name: str = "input"
    is_input: bool = True

    _circuits: List[QuantumCircuit] = None
    _targets: List[int] = None

    def __init__(self, qubit_num: int, input_values: List[List[int]]) -> None:
        self._targets = list(range(qubit_num))
        self._circuits = self.build_circuits(qubit_num, input_values)

    @abstractmethod
    def build_circuits(self, qubit_num: int, input_values: List[List[int]]) -> None: ...

    def mutate_operands(self) -> None:
        pass

    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit:
        encoding_circuit = self._circuits[self._case_index].to_gate(label="input")
        circuit.append(encoding_circuit, self._targets)
        return circuit

    def __repr__(self) -> str:
        return f"{self.name}({','.join(['target' + str((i + 1)) + '=' + str(target) for i, target in enumerate(self._targets)])})"


class InputEncodingConstructor(ABC):
    input_values: List[List[int]] = None

    def __init__(self, input_values: List[List[int]]) -> None:
        self.input_values = input_values

    @abstractmethod
    def __call__(self, qubit_num: int) -> InputEncoding: ...
