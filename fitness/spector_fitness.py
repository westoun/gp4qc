#!/usr/bin/env python3

from scipy.spatial import distance
from statistics import mean
from typing import List, Tuple

from gates import Gate, InputEncoding, Identity
from .fitness import Fitness
from .utils import build_circuit, run_circuit, aggregate_state_distribution
from .params import FitnessParams

class SpectorFitness(Fitness):
    def __init__(
        self, target_distributions: List[List[float]], params: FitnessParams
    ) -> None:
        self.target_distributions = target_distributions

        self.params = params 

    def evaluate(self, chromosome: List[Gate]) -> Tuple[List[Gate], float]:
        hits: int = len(self.target_distributions)
        errors: List[float] = []

        for i, target_distribution in enumerate(self.target_distributions):
            circuit = build_circuit(
                chromosome,
                qubit_num=self.params.qubit_num,
                case_index=i,
            )

            state_distribution = run_circuit(circuit)

            state_distribution = aggregate_state_distribution(
                state_distribution,
                measurement_qubit_num=self.params.measurement_qubit_num,
                ancillary_num=self.params.qubit_num - self.params.measurement_qubit_num,
            )

            assert len(state_distribution) == len(
                target_distribution
            ), f"Missmatch between produced distribution (len {len(state_distribution)}) and target distribution (len {len(target_distribution)})" 

            match_index = target_distribution.index(1.0)
            assert (
                match_index != -1
            ), f"Check the formatting of your target distributions. A 1 is missing in the {i + 1}. distribution."

            probability = state_distribution[match_index]
            if probability >= 0.52:
                hits -= 1 
            else:
                error = distance.jensenshannon(state_distribution, target_distribution)
                errors.append(error)

        fitness_score: float = 0
        if len(errors) > 0:
            fitness_score = hits + sum(errors) / max(hits, 1)
        else:
            fitness_score = len([
                gate for gate in chromosome if type(gate) != Identity
            ]) / 100000

        for validity_check in self.params.validity_checks:
            if not validity_check(chromosome):
                fitness_score += 100
                break

        return chromosome, fitness_score