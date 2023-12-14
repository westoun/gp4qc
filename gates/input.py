#!/usr/bin/env python3

from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from typing import List

from .gate import Gate


class InputEncoding(Gate, ABC):
    """ """

    _input_index: int = 0

    def __init__(self, input_values: List[List[int]], qubit_num: int) -> None:
        self._qubit_num = qubit_num
        self._input_values = input_values

    def mutate_operands(self) -> None:
        pass

    @abstractmethod
    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit:
        ...

    def set_input_index(self, input_index: int) -> "InputEncoding":
        self._input_index = input_index
        return self


class BinaryEncoding(InputEncoding):
    def apply_to(
        self, circuit: QuantumCircuit, input_index: int = None
    ) -> QuantumCircuit:
        if input_index is not None:
            self.set_input_index(input_index)

        for i, qubit_value in enumerate(self._input_values[self._input_index]):
            if qubit_value == 0:
                continue
            elif qubit_value == 1:
                circuit.x(i)
            else:
                raise NotImplementedError()

        return circuit
