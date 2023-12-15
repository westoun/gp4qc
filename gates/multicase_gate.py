#!/usr/bin/env pyhton3

from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from typing import List

from .gate import Gate


class MultiCaseGate(Gate):
    """Base class for gates that act differently depending
    on the case it is run on. Oracles and input gates, for
    example, return different circuits depending on the
    test case.
    """

    _case_index: int = 0

    def set_case_index(self, index: int) -> "MultiCaseGate":
        self._case_index = index
        return self
