#!/usr/bin/env python3

import numpy as np
from quasim import Circuit
from quasim.gates import CRY as CRYGate
from random import random, sample
from typing import List, Union, Tuple

from .optimizable_gate import OptimizableGate


class CRY(OptimizableGate):
    name: str = "cry"

    control: int
    target: int
    theta: float

    def __init__(self, qubit_num: int):
        assert (
            qubit_num > 1
        ), "The CRY Gate requires at least 2 qubits to operate as intended."

        self._qubit_num = qubit_num
        self.target, self.control = sample(range(0, qubit_num), 2)

        # Choose theta randomly, since theta = 0 is often a stationary
        # point and fails numerical optimizers to progress.
        self.theta = random() * 2 * np.pi - np.pi

    def mutate_operands(self) -> None:
        self.target, self.control = sample(range(0, self._qubit_num), 2)

    def apply_to(self, circuit: Circuit) -> Circuit:
        circuit.apply(CRYGate(self.control, self.target, theta=self.theta))
        return circuit

    def __repr__(self) -> str:
        return f"{self.name}(theta={self.theta},control={self.control},target={self.target})"

    @property
    def params(self) -> List[float]:
        return [self.theta]

    @property
    def bounds(self) -> List[Union[Tuple[float, float], None]]:
        return [(-np.pi, np.pi)]

    def set_params(self, params: List[float]) -> None:
        assert len(params) == 1, "The CRY gate requires exactly one parameter!"

        self.theta = params[0]
