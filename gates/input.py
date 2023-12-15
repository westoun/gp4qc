#!/usr/bin/env python3

from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from typing import List

from .multicase_gate import MultiCaseGate


class InputEncoding(MultiCaseGate, ABC):
    """ """

    def __init__(self, input_values: List[List[int]], qubit_num: int) -> None:
        self._qubit_num = qubit_num
        self._input_values = input_values

    def mutate_operands(self) -> None:
        pass

    @abstractmethod
    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit:
        ...


class BinaryEncoding(InputEncoding):
    def apply_to(
        self, circuit: QuantumCircuit, case_index: int = None
    ) -> QuantumCircuit:
        if case_index is not None:
            self.set_case_index(case_index)

        for i, qubit_value in enumerate(self._input_values[self._case_index]):
            if qubit_value == 0:
                continue
            elif qubit_value == 1:
                circuit.x(i)
            else:
                raise NotImplementedError()

        return circuit
