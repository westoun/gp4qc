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
from fitness import Fitness, Jensensshannon, build_circuit


def construct_oracle_circuit(
    input_values: List[List[int]],
    target_distributions: List[List[float]],
    qubit_num: int = 2,
) -> QuantumCircuit:
    ga_params = GAParams(
        population_size=100,
        generations=40,
        crossover_prob=0.5,
        swap_mutation_prob=0.3,
        operand_mutation_prob=0.1,
        chromosome_length=4 + 1,  # + 1 for input gate
        fitness_threshold=0.1,
        fitness_threshold_at=1,
        log_average_fitness=False,
    )
    gate_set: GateSet = GateSet(
        gates=[
            Hadamard,
            X,
            CX,
            Identity,
            BinaryEncoding.init_circuits(input_values, qubit_num),
        ],
        qubit_num=qubit_num,
    )

    fitness: Fitness = Jensensshannon(
        target_distributions=target_distributions, qubit_num=qubit_num
    )

    genetic_algorithm = GA(gate_set, fitness, params=ga_params)
    genetic_algorithm.run()

    chromosome, fitness_value = genetic_algorithm.get_best_chromosomes(n=1)[0]

    if fitness_value > ga_params.fitness_threshold:
        print("Fitness threshold of oracle not reached.")
        print(f"Using best oracle at fitness of {fitness_value}")
    else:
        print("Successfully learned oracle.")

    circuit = build_circuit(chromosome, qubit_num=qubit_num)
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

    QUBIT_NUM = 2
    MEASUREMENT_QUBIT_NUM = 1

    balanced_equal_oracle_circuit = construct_oracle_circuit(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [
            encode([0, 0]),
            encode([0, 1]),
            encode([1, 1]),
            encode([1, 0]),
        ],
        qubit_num=QUBIT_NUM,
    )
    balanced_swapped_oracle_circuit = construct_oracle_circuit(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [
            encode([0, 1]),
            encode([0, 0]),
            encode([1, 0]),
            encode([1, 1]),
        ],
        qubit_num=QUBIT_NUM,
    )
    constant_0_oracle_circuit = construct_oracle_circuit(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [
            encode([0, 0]),
            encode([0, 1]),
            encode([1, 0]),
            encode([1, 1]),
        ],
        qubit_num=QUBIT_NUM,
    )
    constant_1_oracle_circuit = construct_oracle_circuit(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        [
            encode([0, 1]),
            encode([0, 0]),
            encode([1, 1]),
            encode([1, 0]),
        ],
        qubit_num=QUBIT_NUM,
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
            BinaryEncoding.init_circuits(input_values, qubit_num=QUBIT_NUM),
        ],
        qubit_num=QUBIT_NUM,
    )

    fitness: Fitness = Jensensshannon(
        target_distributions=target_distributions,
        qubit_num=QUBIT_NUM,
        measurement_qubit_num=MEASUREMENT_QUBIT_NUM,
    )

    genetic_algorithm = GA(gate_set, fitness, params=ga_params)
    genetic_algorithm.run()

    TOP_N = 3
    for chromosome, fitness_value in genetic_algorithm.get_best_chromosomes(n=TOP_N):
        circuit = build_circuit(chromosome, qubit_num=QUBIT_NUM)

        print(f"\nFitness value: {fitness_value}")
        print(circuit)

        # circuit.draw("mpl")
        # plt.show()
