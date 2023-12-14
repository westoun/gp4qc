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
        errors: List[float] = []

        for i, target_distribution in enumerate(self.target_distributions):
            circuit = build_circuit(
                chromosome,
                qubit_num=self.qubit_num,
                input_gate=self.input_gate,
                case_index=i,
            )

            state_distribution = run_circuit(circuit)

            state_distribution = aggregate_state_distribution(
                state_distribution,
                measurement_qubit_num=self.measurement_qubit_num,
                ancillary_num=self.qubit_num - self.measurement_qubit_num,
            )

            assert len(state_distribution) == len(
                target_distribution
            ), f"Missmatch between produced distribution (len {len(state_distribution)}) and target distribution (len {len(target_distribution)})"

            error = distance.jensenshannon(state_distribution, target_distribution)
            errors.append(error)

        error = mean(errors)
        return (error,)
