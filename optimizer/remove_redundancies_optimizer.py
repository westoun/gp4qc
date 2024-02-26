#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Tuple

from fitness import Fitness
from gates import Gate, Identity, GateSet
from .params import OptimizerParams, default_params
from .optimizer import Optimizer
from .utils import build_circuit, run_circuit, aggregate_state_distribution

class RemoveRedundanciesOptimizer(Optimizer):
    def __init__(
        self, target_distributions: List[List[float]], params: OptimizerParams = default_params
    ) -> None:
        self.target_distributions = target_distributions
        self.params = params

    def optimize(self, chromosome: List[Gate], fitness: Fitness) -> Tuple[List[Gate], float]:
        for i in range(len(chromosome) - 1):
            gate = chromosome[i]
            next_gate = chromosome[i + 1]

            if gate == next_gate and not type(gate) == Identity:
                chromosome[i] = Identity(qubit_num=self.params.qubit_num)
                chromosome[i + 1] = Identity(qubit_num=self.params.qubit_num)
        
        state_distributions: List[List[float]] = []

        for i in range(len(self.target_distributions)):
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

            state_distributions.append(state_distribution)

        fitness_score = fitness.evaluate(state_distributions, self.target_distributions, chromosome)

        return chromosome, fitness_score