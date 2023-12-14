#!/usr/bin/env python3

from scipy.spatial import distance
from statistics import mean
from typing import List

from gates import Gate, InputEncoding
from .fitness import Fitness
from .utils import build_circuit, run_circuit, aggregate_state_distribution


class Jensensshannon(Fitness):
    def __init__(
        self,
        target_distributions: List[List[float]],
        qubit_num: int,
        ancillary_num: int = 0,
        input_gate: InputEncoding = None,
    ) -> None:
        self.target_distributions = target_distributions
        self.qubit_num = qubit_num
        self.ancillary_num = ancillary_num
        self.input_gate = input_gate

    def evaluate(self, chromosome: List[Gate]) -> float:
        errors: List[float] = []

        for i, target_distribution in enumerate(self.target_distributions):
            if self.input_gate is not None:
                self.input_gate.set_input_index(i)

            circuit = build_circuit(
                chromosome, qubit_num=self.qubit_num + self.ancillary_num, input_gate=self.input_gate
            )
            state_distribution = run_circuit(circuit)

            if self.ancillary_num > 0:
                state_distribution = aggregate_state_distribution(
                    state_distribution, self.qubit_num, self.ancillary_num
                )

            assert len(state_distribution) == len(target_distribution)

            error = distance.jensenshannon(state_distribution, target_distribution)
            errors.append(error)

        error = mean(errors)
        return (error,)
