#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
from qiskit import QuantumCircuit
from statistics import mean
from typing import List

from gates import (
    Gate,
    GateSet,
    Hadamard,
    CX,
    CY,
    CZ,
    Identity,
    X,
    Y,
    Z,
    Swap,
    CCX,
    CCZ,
    Oracle,
    HadamardLayer,
)
from ga import GA, GAParams
from fitness import Fitness, Jensensshannon, FitnessParams, SpectorFitness
from fitness.validity_checks import uses_oracle, uses_hadamard_layer
from optimizer import Optimizer, DoNothingOptimizer, OptimizerParams, build_circuit


def construct_oracle_circuit(target_state: List[int]) -> QuantumCircuit:
    # Circuit design taken from
    # https://quantumcomputing.stackexchange.com/q/8850

    circuit = QuantumCircuit(len(target_state))

    for i, qubit_state in enumerate(target_state):
        if qubit_state == 0:
            circuit.x(i)

    circuit.ccz(0, 1, 2)

    for i, qubit_state in enumerate(target_state):
        if qubit_state == 0:
            circuit.x(i)

    return circuit


def state_to_distribution(target_state: List[int]) -> List[float]:
    vectors = []
    for qubit_state in target_state:
        if qubit_state == 0:
            vectors.append([1, 0])
        else:
            vectors.append([0, 1])

    distribution = vectors[0]
    for vector in vectors[1:]:
        distribution = np.kron(distribution, vector)

    return distribution.tolist()


def run_grover():
    target_states = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]

    oracle_circuits: List[QuantumCircuit] = []
    for target_state in target_states:
        oracle_circuit = construct_oracle_circuit(target_state)
        oracle_circuits.append(oracle_circuit)

    target_distributions = [state_to_distribution(state) for state in target_states]

    gate_set: GateSet = GateSet(
        gates=[
            Hadamard,
            CX,
            CY,
            CZ,
            Identity,
            X,
            Y,
            Z,
            Swap,
            CCX,
            CCZ,
            Oracle.set_circuits(oracle_circuits),
            HadamardLayer,
        ],
        qubit_num=3,
    )

    ga_params = GAParams(
        population_size=500,
        generations=10,
        crossover_prob=0.4,
        swap_gate_mutation_prob=0.1,
        swap_order_mutation_prob=0.1,
        operand_mutation_prob=0.1,
        chromosome_length=10,
        log_average_fitness_at=1,
    )

    fitness_params = FitnessParams(validity_checks=[uses_oracle, uses_hadamard_layer])
    fitness: Fitness = SpectorFitness(params=fitness_params)

    optimizer_params = OptimizerParams(qubit_num=3, measurement_qubit_num=3)
    optimizer: Optimizer = DoNothingOptimizer(
        target_distributions, params=optimizer_params
    )

    genetic_algorithm = GA(gate_set, fitness, optimizer, params=ga_params)

    average_fitness_values = []

    def log_fitness_callback(
        ga: GA, population: List[List[Gate]], fitness_values: List[float], generation: int
    ) -> None:
        fitness_values = [
            value % 100 for value in fitness_values
        ]  # Remove punishment terms
        average_fitness = mean(fitness_values)
        average_fitness_values.append(average_fitness)

    genetic_algorithm.on_after_generation(log_fitness_callback)
    genetic_algorithm.run()

    plt.plot(average_fitness_values)
    plt.xlabel("generation")
    plt.ylabel("mean fitness")
    plt.savefig("mean_fitness.png")
    plt.show()

    TOP_N = 3
    for chromosome, fitness_value in genetic_algorithm.get_best_chromosomes(n=TOP_N):
        circuit = build_circuit(chromosome, qubit_num=3)

        print(f"\nFitness value: {fitness_value}")
        print(circuit)

        # circuit.draw("mpl")
        # plt.show()
