#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractclassmethod
from qiskit import QuantumCircuit
from typing import List

from .multicase_gate import MultiCaseGate


class InputEncoding(MultiCaseGate, ABC):
    """ """

    _circuits: List[QuantumCircuit] = None
    _targets: List[int] = None

    def __init__(self, qubit_num: int) -> None:
        assert (
            self._circuits is not None
        ), "No circuits have been initialized. Has init_circuits() been called?"

        # the qubit num passed to init may be different from
        # qubit num passed to init_circuits. Especially if
        # ancillary qubits are used.

    @abstractclassmethod
    def init_circuits(
        cls, input_values: List[List[int]], qubit_num: int
    ) -> "InputEncoding":
        ...

    def mutate_operands(self) -> None:
        pass

    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit:
        encoding_circuit = self._circuits[self._case_index].to_gate(label="input")
        circuit.append(encoding_circuit, self._targets)
        return circuit


class BinaryEncoding(InputEncoding):
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
