#!/usr/bin/env python3

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
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
    Swap,
    CCX,
    Identity,
    InputEncoding,
    BinaryEncoding,
    Oracle,
)
from ga import GA, GAParams
from fitness import Fitness, Jensensshannon, build_circuit, FitnessParams


def construct_oracle_circuit(
    input_values: List[List[int]], target_distributions: List[List[float]]
) -> QuantumCircuit:
    ga_params = GAParams(
        population_size=200,
        generations=40,
        crossover_prob=0.5,
        swap_mutation_prob=0.3,
        operand_mutation_prob=0.2,
        chromosome_length=3 + 1,  # + 1 for input gate
        fitness_threshold=0.1,
        fitness_threshold_at=1,
        log_average_fitness=False,
    )
    fitness_params = FitnessParams(qubit_num=2, measurement_qubit_num=2)

    gate_set: GateSet = GateSet(
        gates=[
            X,
            CX,
            Identity,
            BinaryEncoding.init_circuits(input_values, 2),
        ],
        qubit_num=2,
    )

    fitness: Fitness = Jensensshannon(
        target_distributions=target_distributions, params=fitness_params
    )

    genetic_algorithm = GA(gate_set, fitness, params=ga_params)
    genetic_algorithm.run()

    chromosome, fitness_value = genetic_algorithm.get_best_chromosomes(n=1)[0]

    if fitness_value > ga_params.fitness_threshold:
        print("Fitness threshold of oracle not reached.")
        print(f"Using best oracle at fitness of {fitness_value}")
    else:
        print("Successfully learned oracle.")

    circuit = build_circuit(chromosome, qubit_num=2)
    return circuit


def encode(states: List[int]) -> List[int]:
    encoding = []
    for _ in range(2 ** len(states)):
        encoding.append(0)

    encoding_index = 0
    for i, state in enumerate(reversed(states)):
        encoding_index += state * 2**i

    encoding[encoding_index] = 1
    return encoding


def run_deutsch():
    ga_params = GAParams(
        population_size=100,
        generations=50,
        crossover_prob=0.4,
        swap_mutation_prob=0.2,
        operand_mutation_prob=0.3,
        chromosome_length=4 + 1,  # + 1 for encoding layer
        fitness_threshold=0.1,
        fitness_threshold_at=10,
    )
    fitness_params = FitnessParams(qubit_num=2, measurement_qubit_num=1)

    balanced_equal_oracle_circuit = construct_oracle_circuit(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [
            encode([0, 0]),
            encode([0, 1]),
            encode([1, 1]),
            encode([1, 0]),
        ],
    )
    balanced_swapped_oracle_circuit = construct_oracle_circuit(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [
            encode([0, 1]),
            encode([0, 0]),
            encode([1, 0]),
            encode([1, 1]),
        ],
    )
    constant_0_oracle_circuit = construct_oracle_circuit(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [
            encode([0, 0]),
            encode([0, 1]),
            encode([1, 0]),
            encode([1, 1]),
        ],
    )
    constant_1_oracle_circuit = construct_oracle_circuit(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [
            encode([0, 1]),
            encode([0, 0]),
            encode([1, 1]),
            encode([1, 0]),
        ],
    )
    oracle_circuits = [
        balanced_equal_oracle_circuit,
        balanced_swapped_oracle_circuit,
        constant_0_oracle_circuit,
        constant_1_oracle_circuit,
    ]

    input_values: List[List[int]] = [
        [0],
        [0],
        [0],
        [0],
    ]
    target_distributions: List[List[float]] = [
        [0, 1],  # balanced equal
        [0, 1],  # balanced swapped
        [1, 0],  # constant 0
        [1, 0],  # constant 1
    ]

    gate_set: GateSet = GateSet(
        gates=[
            Hadamard,
            X,
            Identity,
            Oracle.set_circuits(oracle_circuits),
            BinaryEncoding.init_circuits(input_values, qubit_num=1),
        ],
        qubit_num=2,
    )

    fitness: Fitness = Jensensshannon(
        target_distributions=target_distributions, params=fitness_params
    )

    genetic_algorithm = GA(gate_set, fitness, params=ga_params)
    genetic_algorithm.run()

    TOP_N = 3
    for chromosome, fitness_value in genetic_algorithm.get_best_chromosomes(n=TOP_N):
        circuit = build_circuit(chromosome, qubit_num=2)

        print(f"\nFitness value: {fitness_value}")
        print(circuit)

        # circuit.draw("mpl")
        # plt.show()
