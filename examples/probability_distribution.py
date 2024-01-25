#!/usr/bin/env python3

import matplotlib.pyplot as plt
from typing import List

from gates import (
    Gate,
    GateSet,
    Hadamard,
    X,
    Y,
    Z,
    CX,
    CY,
    CZ,
    CCX,
    CCZ,
    RX,
    RZ,
    RY,
    Swap,
    CRX,
    CRY,
    CRZ,
    Identity,
    Phase,
)
from ga import GA, GAParams
from fitness import Fitness, Jensensshannon, FitnessParams
from optimizer import (
    Optimizer,
    NumericalOptimizer,
    OptimizerParams,
    build_circuit,
    run_circuit,
)
from fitness.validity_checks import uses_parametrized_gates, uses_hadamard


def run_create_probability_distribution():
    gate_set: GateSet = GateSet(
        gates=[Hadamard, X, Y, Z, CRX, CRY, CRZ, Phase, Swap, Identity], qubit_num=2
    )
    target_distributions: List[List[float]] = [[1 / 6, 1 / 6, 1 / 6, 0.5]]
    ga_params = GAParams(
        population_size=100,
        generations=20,
        crossover_prob=0.4,
        swap_gate_mutation_prob=0.1,
        swap_order_mutation_prob=0.1,
        chromosome_length=4,
        log_average_fitness_at=1,
        fitness_threshold=0.08,
    )

    fitness_params = FitnessParams(
        validity_checks=[uses_parametrized_gates, uses_hadamard]
    )
    fitness: Fitness = Jensensshannon(params=fitness_params)

    optimizer_params = OptimizerParams(
        qubit_num=2, measurement_qubit_num=2, max_iter=20
    )
    optimizer: Optimizer = NumericalOptimizer(
        target_distributions, params=optimizer_params
    )

    genetic_algorithm = GA(gate_set, fitness, optimizer, params=ga_params)
    genetic_algorithm.run()

    TOP_N = 3
    for chromosome, fitness_value in genetic_algorithm.get_best_chromosomes(n=TOP_N):
        circuit = build_circuit(chromosome, qubit_num=2)
        distribution = run_circuit(circuit)

        print(f"\nFitness value: {fitness_value}")
        print(circuit)
        print(distribution)
