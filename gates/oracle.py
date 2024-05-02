#!/usr/bin/env python3

from abc import ABC, abstractmethod
from quasim import Circuit
from random import sample
from typing import Any, List

from .multicase_gate import MultiCaseGate


class Oracle(MultiCaseGate, ABC):
    name: str = "oracle"
    is_oracle: bool = True

    _circuits: List[Circuit] = None
    _oracle_qubit_num: int = None

    targets = []

    def __init__(
        self, qubit_num: int, circuits: List[Circuit], oracle_qubit_num: int
    ) -> None:
        self._circuits = circuits
        self._oracle_qubit_num = oracle_qubit_num

        self._qubit_num = qubit_num

        self.targets = sample(range(0, self._qubit_num), self._oracle_qubit_num)

    def mutate_operands(self) -> None:
        self.targets = sample(range(0, self._qubit_num), self._oracle_qubit_num)

    def apply_to(self, circuit: Circuit) -> Circuit:
        for gate in self._circuits[self._case_index].gates:
            circuit.apply(gate)
        return circuit

    def __repr__(self) -> str:
        return f"{self.name}({','.join(['target' + str((i + 1)) + '=' + str(target) for i, target in enumerate(self.targets)])})"


class OracleConstructor:
    _circuits: List[Circuit] = None
    _oracle_qubit_num: int = None

    def __init__(self, circuits: List[Circuit]) -> None:
        self._circuits = circuits
        self._oracle_qubit_num = circuits[0].qubit_num

    def __call__(self, qubit_num: int) -> Oracle:
        return Oracle(
            qubit_num=qubit_num,
            circuits=self._circuits,
            oracle_qubit_num=self._oracle_qubit_num,
        )
