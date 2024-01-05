#!/usr/bin/env python3

from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from random import sample
from typing import List

from .multicase_gate import MultiCaseGate


class Oracle(MultiCaseGate, ABC):
    """ """

    _circuits: List[QuantumCircuit] = None
    _oracle_qubit_num: int = None

    targets = []

    def __init__(
        self,
        qubit_num: int,
    ) -> None:
        assert self._circuits is not None, "No circuits have been provided."
        assert self._oracle_qubit_num is not None

        self._qubit_num = qubit_num

        self.targets = sample(range(0, self._qubit_num), self._oracle_qubit_num)

    def mutate_operands(self) -> None:
        self.targets = sample(range(0, self._qubit_num), self._oracle_qubit_num)

    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit:
        oracle_circuit = self._circuits[self._case_index].to_gate(label="oracle")
        circuit.append(oracle_circuit, self.targets)
        return circuit

    @classmethod
    def set_circuits(cls, circuits: List[QuantumCircuit]) -> "Oracle":
        cls._circuits = circuits
        cls._oracle_qubit_num = len(circuits[0].qubits)
        return cls

    def __repr__(self) -> str:
        return f"oracle({','.join(['target' + str((i + 1)) + '=' + str(target) for i, target in enumerate(self.targets)])})"
