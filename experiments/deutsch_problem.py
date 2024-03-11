#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
from qiskit import QuantumCircuit
from statistics import mean
from typing import List
from uuid import uuid4

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
    HadamardLayer,
    Identity,
    InputEncoding,
    BinaryEncoding,
    InputEncodingConstructor,
    Oracle,
    OracleConstructor,
)
from ga import GA, GAParams
from fitness import Fitness, Jensensshannon, FitnessParams
from fitness.validity_checks import (
    has_exactly_1_input,
    has_exactly_1_oracle,
    has_input_at_first_position,
)
from optimizer import (
    Optimizer,
    DoNothingOptimizer,
    RemoveRedundanciesOptimizer,
    OptimizerParams,
    build_circuit,
)
from utils.logging import (
    log_experiment_details,
    log_fitness,
)

# Place experiment id creation outside of main function
# to avoid having to pass it through multiple layer of
# nested function calls.
EXPERIMENT_ID = f"deutsch_{uuid4()}"


def construct_oracle_circuit(
    input_values: List[List[int]], target_distributions: List[List[float]]
) -> QuantumCircuit:
    ga_params = GAParams(
        population_size=400,
        generations=100,
        crossover_prob=0.5,
        swap_gate_mutation_prob=0.1,
        swap_order_mutation_prob=0.1,
        operand_mutation_prob=0.1,
        chromosome_length=5 + 1,  # + 1 for input gate
        fitness_threshold=0,
        log_average_fitness=False,
    )

    gate_set: GateSet = GateSet(
        gates=[
            X,
            CX,
            Identity,
            InputEncodingConstructor(
                input_values=input_values, EncodingType=BinaryEncoding
            ),
        ],
        qubit_num=2,
    )

    fitness_params = FitnessParams(
        validity_checks=[has_exactly_1_input, has_input_at_first_position],
    )
    fitness: Fitness = Jensensshannon(params=fitness_params)

    optimizer_params = OptimizerParams(qubit_num=2, measurement_qubit_num=2)
    optimizer: Optimizer = DoNothingOptimizer(
        target_distributions, params=optimizer_params
    )

    genetic_algorithm = GA(gate_set, fitness, optimizer, params=ga_params)
    genetic_algorithm.run()

    chromosome, fitness_value = genetic_algorithm.get_best_chromosomes(n=1)[0]

    if fitness_value > ga_params.fitness_threshold:
        print("Fitness threshold of oracle not reached.")
        print(f"Using best oracle at fitness of {fitness_value}")
    else:
        print("Successfully learned oracle.")

    # Remove input layer
    chromosome = chromosome[1:]

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


def create_oracle_circuits():
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
    return oracle_circuits


def run_deutsch():
    # oracle_circuits = create_oracle_circuits()
    # with open("results/deutsch_oracle.pickle", "wb") as oracle_file:
    #     pickle.dump(oracle_circuits, oracle_file)

    with open("results/deutsch_oracle.pickle", "rb") as oracle_file:
        oracle_circuits = pickle.load(oracle_file)

    ga_params = GAParams(
        population_size=1000,
        generations=500,
        crossover_prob=0.5,
        swap_gate_mutation_prob=0.1,
        swap_order_mutation_prob=0.1,
        operand_mutation_prob=0.1,
        chromosome_length=6,
        fitness_threshold=0.01,
    )

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
            Y,
            Z,
            CX,
            CY,
            CZ,
            Swap,
            HadamardLayer,
            Identity,
            OracleConstructor(oracle_circuits),
        ],
        qubit_num=2,
    )

    fitness_params = FitnessParams(
        validity_checks=[
            has_exactly_1_oracle,
        ],
    )
    fitness: Fitness = Jensensshannon(params=fitness_params)

    optimizer_params = OptimizerParams(qubit_num=2, measurement_qubit_num=1)
    optimizer: Optimizer = RemoveRedundanciesOptimizer(
        target_distributions, params=optimizer_params
    )

    genetic_algorithm = GA(gate_set, fitness, optimizer, params=ga_params)

    log_experiment_details(
        ga=genetic_algorithm,
        experiment_id=EXPERIMENT_ID,
        target_path="results/experiments.csv",
    )

    def log_fitness_callback(
        ga: GA,
        population: List[List[Gate]],
        fitness_values: List[float],
        generation: int,
    ) -> None:
        best_chromosome, best_fitness_value = ga.get_best_chromosomes(1)[0]
        mean_fitness_value = mean(fitness_values)

        log_fitness(
            experiment_id=EXPERIMENT_ID,
            generation=generation,
            best_fitness_value=best_fitness_value,
            mean_fitness_value=mean_fitness_value,
            best_chromosome=best_chromosome,
            target_path="results/fitness_values.csv",
        )

    genetic_algorithm.on_after_generation(log_fitness_callback)

    genetic_algorithm.run()

    TOP_N = 3
    for chromosome, fitness_value in genetic_algorithm.get_best_chromosomes(n=TOP_N):
        circuit = build_circuit(chromosome, qubit_num=2)

        print(f"\nFitness value: {fitness_value}")
        print(chromosome)
        print(circuit)

        # circuit.draw("mpl")
        # plt.show()
