#!/usr/bin/env python3

import matplotlib.pyplot as plt
from typing import List

from gates import Gate, GateSet, Hadamard, X, Y, Z, CX, CY, CZ, Identity
from ga import GA, GAParams
from fitness import Fitness, Jensensshannon, build_circuit, FitnessParams


def run_bell_state_3qubits():
    gate_set: GateSet = GateSet(
        gates=[Hadamard, X, Y, Z, CX, CY, CZ, Identity], qubit_num=3
    )
    target_distributions: List[List[float]] = [[0.5, 0, 0, 0, 0, 0, 0, 0.5]]

    ga_params = GAParams(
        population_size=100,
        generations=50,
        crossover_prob=0.5,
        swap_mutation_prob=0.1,
        operand_mutation_prob=0.4,
        chromosome_length=5,
        fitness_threshold=0.01,
        fitness_threshold_at=3,
        log_average_fitness_at=1,
    )
    fitness_params = FitnessParams(qubit_num=3, measurement_qubit_num=3)

    fitness: Fitness = Jensensshannon(
        target_distributions=target_distributions, params=fitness_params
    )

    genetic_algorithm = GA(gate_set, fitness, params=ga_params)
    genetic_algorithm.run()

    TOP_N = 3
    for chromosome, fitness_value in genetic_algorithm.get_best_chromosomes(n=TOP_N):
        circuit = build_circuit(chromosome, qubit_num=3)

        print(f"\nFitness value: {fitness_value}")
        print(circuit)

        # circuit.draw("mpl")
        # plt.show()
