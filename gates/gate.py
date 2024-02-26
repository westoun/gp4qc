#!/usr/bin/env pyton3

from abc import ABC, abstractmethod, abstractclassmethod
from qiskit import QuantumCircuit


class Gate(ABC):
    """Base class of all gates including functions for
    their mutation behavior and circuit representation.
    """

    @abstractmethod
    def __init__(self, qubit_num: int) -> None:
        ...

    @abstractmethod
    def mutate_operands(self) -> None:
        ...

    @abstractmethod
    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit:
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: "Gate") -> bool:
        return self.__repr__() == other.__repr__()

    @property
    @abstractclassmethod
    def name() -> str:
        ...
