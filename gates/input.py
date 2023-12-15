#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractclassmethod
from qiskit import QuantumCircuit
from typing import List

from .multicase_gate import MultiCaseGate


class InputEncoding(MultiCaseGate, ABC):
    """ """

    _circuits: List[QuantumCircuit] = None

    def __init__(self, qubit_num: int) -> None:
        assert (
            self._circuits is not None
        ), "No circuits have been initialized. Has init_circuits() been called?"

        self._qubit_num = qubit_num

    @abstractclassmethod
    def init_circuits(
        cls, input_values: List[List[int]], qubit_num: int, measurement_qubit_num: int
    ) -> "InputEncoding":
        ...

    def mutate_operands(self) -> None:
        pass

    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit:
        encoding_circuit = self._circuits[self._case_index].to_gate(label="input")
        targets = list(range(self._qubit_num))
        circuit.append(encoding_circuit, targets)
        return circuit


class BinaryEncoding(InputEncoding):
    @classmethod
    def init_circuits(
        cls, input_values: List[List[int]], qubit_num: int
    ) -> "BinaryEncoding":
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
