#!/usr/bin/env python3

from scipy.spatial import distance
from statistics import mean
from typing import List

from gates import Gate
from .fitness import Fitness
from .utils import build_circuit, run_circuit


class Jensensshannon(Fitness):
    def __init__(self, target_distributions: List[List[float]], qubit_num: int) -> None:
        self.target_distributions = target_distributions
        self.qubit_num = qubit_num

    def evaluate(self, chromosome: List[Gate]) -> float:
        circuit = build_circuit(chromosome, self.qubit_num)

        errors: List[float] = []
        for target_distribution in self.target_distributions:
            state_distribution = run_circuit(circuit)

            assert len(state_distribution) == len(target_distribution)

            error = distance.jensenshannon(state_distribution, target_distribution)
            errors.append(error)

        error = mean(errors)
        return (error,)
