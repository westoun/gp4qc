#!/usr/bin/env pyton3

from abc import ABC, abstractmethod
from qiskit import QuantumCircuit


class Gate(ABC):
    @abstractmethod
    def __init__(self, qubit_num: int) -> None:
        ...

    @abstractmethod
    def mutate_operands(self) -> None:
        ...

    @abstractmethod
    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit:
        ...
