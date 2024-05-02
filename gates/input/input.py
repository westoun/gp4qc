#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractclassmethod
from quasim import Circuit
from typing import Any, List, Type

from gates.multicase_gate import MultiCaseGate


class InputEncoding(MultiCaseGate, ABC):
    name: str = "input"
    is_input: bool = True

    _circuits: List[Circuit] = None
    _targets: List[int] = None

    def __init__(self, qubit_num: int, input_values: List[List[int]]) -> None:
        self._targets = list(range(qubit_num))
        self._circuits = self.build_circuits(qubit_num, input_values)

    @abstractmethod
    def build_circuits(self, qubit_num: int, input_values: List[List[int]]) -> None: ...

    def mutate_operands(self) -> None:
        pass

    def apply_to(self, circuit: Circuit) -> Circuit:
        for gate in self._circuits[self._case_index].gates:
            circuit.apply(gate)
        return circuit

    def __repr__(self) -> str:
        return f"{self.name}({','.join(['target' + str((i + 1)) + '=' + str(target) for i, target in enumerate(self._targets)])})"

    @property
    def gate_count(self) -> int:
        return len(self._targets)


class InputEncodingConstructor(ABC):
    input_values: List[List[int]]
    EncodingType: Type

    def __init__(self, input_values: List[List[int]], EncodingType: Type) -> None:
        self.input_values = input_values
        self.EncodingType = EncodingType

    def __call__(self, qubit_num: int) -> InputEncoding:
        return self.EncodingType(qubit_num, self.input_values)
