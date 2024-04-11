#!/usr/bin/env python3

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import warnings

from gates import (
    Gate,
    GateSet,
    H,
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


def compute_gate_correlation_every(
    ga: GA,
    population: List[List[Gate]],
    fitness_values: List[float],
    generation: int,
    n=5,
) -> None:
    if generation % n != 0:
        return

    fitness_values = [
        value % 100 for value in fitness_values
    ]  # Remove punishment terms

    gate_indicators = {}
    for gate in ga.gate_set.gates:
        gate_indicators[gate.name] = []

    for chromosome in population:
        encountered_gates = [gate.name for gate in chromosome]
        encountered_gates = set(encountered_gates)

        for key in gate_indicators:
            if key in encountered_gates:
                gate_indicators[key].append(1)
            else:
                gate_indicators[key].append(0)

    print(f"\nCorrelations after generation {generation}:")
    for key in gate_indicators:
        if np.std(gate_indicators[key]) == 0:
            if gate_indicators[key][0] == 0:
                print(f"\t{key}: NAN (not present in any chromosome)")
            else:
                print(f"\t{key}: NAN (present in every chromosome)")
        else:
            correlation = np.corrcoef(gate_indicators[key], fitness_values)[0, 1]
            print(f"\t{key}: {correlation}")


compute_gate_correlation = partial(compute_gate_correlation_every, n=1)
compute_gate_correlation_every_5 = partial(compute_gate_correlation_every, n=5)


def run_create_probability_distribution():
    gate_set: GateSet = GateSet(
        gates=[
            H,
            X,
            Y,
            Z,
            CX,
            CY,
            RX,
            RZ,
            RY,
            CZ,
            CRX,
            CRY,
            CRZ,
            Phase,
            Swap,
            Identity,
        ],
        qubit_num=2,
    )
    target_distributions: List[List[float]] = [[0, 0.25, 0.25, 0.5]]
    ga_params = GAParams(
        population_size=200,
        generations=20,
        crossover_prob=0.4,
        swap_gate_mutation_prob=0.2,
        swap_order_mutation_prob=0.1,
        chromosome_length=5,
        log_average_fitness_at=1,
        fitness_threshold=0.001,
        fitness_threshold_at=3,
    )

    fitness_params = FitnessParams(
        # validity_checks=[uses_parametrized_gates, uses_hadamard]
    )
    fitness: Fitness = Jensensshannon(params=fitness_params)

    optimizer_params = OptimizerParams(
        qubit_num=2, measurement_qubit_num=2, max_iter=20
    )
    optimizer: Optimizer = NumericalOptimizer(
        target_distributions, params=optimizer_params
    )

    genetic_algorithm = GA(gate_set, fitness, optimizer, params=ga_params)

    genetic_algorithm.on_after_generation(compute_gate_correlation_every_5)
    genetic_algorithm.on_completion(compute_gate_correlation)
    genetic_algorithm.run()

    TOP_N = 3
    for chromosome, fitness_value in genetic_algorithm.get_best_chromosomes(n=TOP_N):
        circuit = build_circuit(chromosome, qubit_num=2)
        distribution = run_circuit(circuit)

        print(f"\nFitness value: {fitness_value}")
        print(circuit)
        print(distribution)
