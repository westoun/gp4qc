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
    measurement_qubit_num: int = 1
) -> QuantumCircuit:
    input_gate: InputEncoding = BinaryEncoding(input_values, qubit_num=qubit_num)

    ga_params = GAParams(
        population_size=100,
        generations=20,
        crossover_prob=0.5,
        swap_mutation_prob=0.1,
        operand_mutation_prob=0.4,
        chromosome_length=4,
        fitness_threshold=0.01,
        fitness_threshold_at=1,
        log_average_fitness=False,
    )
    gate_set: GateSet = GateSet(
        gates=[Hadamard, X, CX, Identity],
        qubit_num=qubit_num,
    )

    fitness: Fitness = Jensensshannon(
        target_distributions=target_distributions,
        qubit_num=qubit_num,
        measurement_qubit_num=measurement_qubit_num,
        input_gate=input_gate,
    )

    genetic_algorithm = GA(gate_set, fitness, params=ga_params)
    genetic_algorithm.run()

    chromosome, fitness_value = genetic_algorithm.get_best_chromosomes(n=1)[0]

    if fitness_value > ga_params.fitness_threshold:
        print("Fitness threshold of oracle not reached.")
        print(f"Using best oracle at fitness of {fitness_value}")

    circuit = build_circuit(chromosome, qubit_num=qubit_num)
    return circuit


def run_deutsch():
    ga_params = GAParams(
        population_size=100,
        generations=50,
        crossover_prob=0.4,
        swap_mutation_prob=0.1,
        operand_mutation_prob=0.1,
        chromosome_length=4,
        fitness_threshold=0.1,
        fitness_threshold_at=1,
    )

    QUBIT_NUM = 2
    MEASUREMENT_QUBIT_NUM = 1

    balanced_equal_oracle_circuit = construct_oracle_circuit(
        [[0], [1]],
        [[1, 0], [0, 1]],
        qubit_num=QUBIT_NUM,
        measurement_qubit_num=MEASUREMENT_QUBIT_NUM,
    )
    balanced_swapped_oracle_circuit = construct_oracle_circuit(
        [[0], [1]],
        [[0, 1], [1, 0]],
        qubit_num=QUBIT_NUM,
        measurement_qubit_num=MEASUREMENT_QUBIT_NUM,
    )
    constant_0_oracle_circuit = construct_oracle_circuit(
        [[0], [1]],
        [[1, 0], [1, 0]],
        qubit_num=QUBIT_NUM,
        measurement_qubit_num=MEASUREMENT_QUBIT_NUM,
    )
    constant_1_oracle_circuit = construct_oracle_circuit(
        [[0], [1]],
        [[0, 1], [0, 1]],
        qubit_num=QUBIT_NUM,
        measurement_qubit_num=MEASUREMENT_QUBIT_NUM,
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

    Oracle.set_circuits(oracle_circuits)

    gate_set: GateSet = GateSet(
        gates=[Hadamard, X, Identity, Oracle],
        qubit_num=QUBIT_NUM,
    )

    input_gate: InputEncoding = BinaryEncoding(input_values, qubit_num=QUBIT_NUM)

    fitness: Fitness = Jensensshannon(
        target_distributions=target_distributions,
        qubit_num=QUBIT_NUM,
        measurement_qubit_num=MEASUREMENT_QUBIT_NUM,
        input_gate=input_gate,
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
