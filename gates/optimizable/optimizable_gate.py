#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractproperty
from qiskit import QuantumCircuit
from typing import List, Tuple, Union

from gates.gate import Gate


class OptimizableGate(Gate, ABC):
    @abstractproperty
    def params(self) -> List[float]:
        ...

    @abstractproperty
    def bounds(self) -> List[Union[Tuple[float, float], None]]:
        ...

    @abstractmethod
    def set_params(self, params: List[float]) -> None:
        ...

    @property
    def param_count(self) -> int:
        return len(self.params)
