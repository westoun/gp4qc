#!/usr/bin/env python3

from scipy.spatial import distance
from statistics import mean
from typing import List

from gates import Gate, InputEncoding
from .fitness import Fitness
from .utils import build_circuit, run_circuit, aggregate_state_distribution


class MatchCount(Fitness):
    def __init__(
        self,
        target_distributions: List[List[float]],
        qubit_num: int,
        measurement_qubit_num: int = None,
        input_gate: InputEncoding = None,
    ) -> None:
        self.target_distributions = target_distributions
        self.qubit_num = qubit_num

        if measurement_qubit_num is None:
            self.measurement_qubit_num = qubit_num
        else:
            self.measurement_qubit_num = measurement_qubit_num

        self.input_gate = input_gate

    def evaluate(self, chromosome: List[Gate]) -> float:
        match_count: int = 0

        for i, target_distribution in enumerate(self.target_distributions):
            if self.input_gate is not None:
                self.input_gate.set_input_index(i)

            circuit = build_circuit(
                chromosome,
                qubit_num=self.qubit_num,
                input_gate=self.input_gate,
            )

            state_distribution = run_circuit(circuit)

            state_distribution = aggregate_state_distribution(
                state_distribution,
                measurement_qubit_num=self.measurement_qubit_num,
                ancillary_num=self.qubit_num - self.measurement_qubit_num,
            )

            assert len(state_distribution) == len(target_distribution)

            match_index = target_distribution.index(1.0)
            assert (
                match_index != -1
            ), f"Check the formatting of your target distributions. A 1 is missing in the {i + 1}. distribution."

            probability = state_distribution[match_index]
            if probability > 0.5:
                match_count += 1

        error = (len(self.target_distributions) - match_count) / len(
            self.target_distributions
        )
        return (error,)
