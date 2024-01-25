#!/usr/bin/env python3

import numpy as np
from qiskit import QuantumCircuit
from random import randint, random
from typing import List, Union, Tuple

from .optimizable_gate import OptimizableGate


class RZ(OptimizableGate):
    target: int
    theta: float

    def __init__(self, qubit_num: int):
        self._qubit_num = qubit_num
        self.target = randint(0, qubit_num - 1)

        # Choose theta randomly, since theta = 0 is often a stationary
        # point and fails numerical optimizers to progress.
        self.theta = random() * 2 * np.pi - np.pi

    def mutate_operands(self) -> None:
        self.target = randint(0, self._qubit_num - 1)

    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit:
        circuit.rz(self.theta, self.target)
        return circuit

    def __repr__(self) -> str:
        return f"rz(theta={self.theta},target={self.target})"

    @property
    def params(self) -> List[float]:
        return [self.theta]

    @property
    def bounds(self) -> List[Union[Tuple[float, float], None]]:
        return [(-np.pi, np.pi)]

    def set_params(self, params: List[float]) -> None:
        assert len(params) == 1, "The RZ gate requires exactly one parameter!"

        self.theta = params[0]
