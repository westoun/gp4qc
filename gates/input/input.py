#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractclassmethod
from qiskit import QuantumCircuit
from typing import List

from gates.multicase_gate import MultiCaseGate


class InputEncoding(MultiCaseGate, ABC):
    name: str = "input" 
    is_input: bool = True

    _circuits: List[QuantumCircuit] = None
    _targets: List[int] = None

    def __init__(self, qubit_num: int) -> None:
        assert (
            self._circuits is not None
        ), "No circuits have been initialized. Has init_circuits() been called?"

        # This re-assignment is necessary to make variables
        # set through cls available as instance variables in
        # a multiprocessing setup.
        self._circuits = self._circuits
        self._targets = self._targets

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

    def __repr__(self) -> str:
        return f"{self.name}({','.join(['target' + str((i + 1)) + '=' + str(target) for i, target in enumerate(self._targets)])})"

