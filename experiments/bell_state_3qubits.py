#!/usr/bin/env python3

import matplotlib.pyplot as plt
from typing import List

from gates import Gate, GateSet, H, X, Y, Z, CX, CY, CZ, Identity
from ga import GA, GAParams
from fitness import Fitness, Jensensshannon, FitnessParams
from optimizer import Optimizer, DoNothingOptimizer, OptimizerParams, build_circuit


def run_bell_state_3qubits():
    gate_set: GateSet = GateSet(
        gates=[H, X, Y, Z, CX, CY, CZ, Identity], qubit_num=3
    )
    target_distributions: List[List[float]] = [[0.5, 0, 0, 0, 0, 0, 0, 0.5]]

    ga_params = GAParams(
        population_size=100,
        generations=50,
        crossover_prob=0.4,
        swap_gate_mutation_prob=0.2,
        swap_order_mutation_prob=0.1,
        operand_mutation_prob=0.1,
        chromosome_length=4,
        fitness_threshold=0.01,
        log_average_fitness_at=1,
    )

    fitness: Fitness = Jensensshannon(
    )

    optimizer_params = OptimizerParams(qubit_num=3, measurement_qubit_num=3)
    optimizer: Optimizer = DoNothingOptimizer(target_distributions, params = optimizer_params)

    genetic_algorithm = GA(gate_set, fitness, optimizer, params=ga_params)
    genetic_algorithm.run()

    TOP_N = 3
    for chromosome, fitness_value in genetic_algorithm.get_best_chromosomes(n=TOP_N):
        circuit = build_circuit(chromosome, qubit_num=3)

        print(f"\nFitness value: {fitness_value}")
        print(circuit)

        # circuit.draw("mpl")
        # plt.show()
