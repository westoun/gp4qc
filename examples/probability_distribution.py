#!/usr/bin/env python3

import matplotlib.pyplot as plt
from typing import List

from gates import Gate, GateSet, Hadamard, X, Y, Z, CX, CY, CZ, CCX, \
    CCZ, Swap, Identity
from ga import GA, GAParams
from fitness import Fitness, Jensensshannon, build_circuit, FitnessParams


def run_create_probability_distribution():
    gate_set: GateSet = GateSet(
        gates=[Hadamard, X, Y, Z, CX, CY, CZ, CCX, CCZ, Swap, Identity], qubit_num=3
    )
    target_distributions: List[List[float]] = [[1/6, 1/6, 1/6, 0.5]]
    ga_params = GAParams(
        population_size=300,
        generations=100,
        crossover_prob=0.4,
        swap_gate_mutation_prob=0.1,
        swap_order_mutation_prob=0.1,
        chromosome_length=6,
        log_average_fitness_at=5,
        fitness_threshold=0.1
    )
    fitness_params = FitnessParams(qubit_num=3, measurement_qubit_num=2)

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