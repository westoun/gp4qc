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
    OracleConstructor,
    HadamardLayer,
    XLayer,
    YLayer,
    ZLayer,
)
from ga import GA, GAParams
from fitness import Fitness, Jensensshannon, FitnessParams, SpectorFitness
from fitness.validity_checks import uses_oracle, uses_hadamard_layer
from optimizer import (
    Optimizer,
    DoNothingOptimizer,
    OptimizerParams,
    build_circuit,
    RemoveRedundanciesOptimizer,
)


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


def compute_bigram_correlations(
    ga: GA,
    population: List[List[Gate]],
    fitness_values: List[float],
    generation: int,
) -> None:
    if generation % 3 != 0:
        return

    population = [ga.toolbox.clone(ind) for ind in population]

    fitness_values = [
        value % 100 for value in fitness_values
    ]  # Remove punishment terms

    bigrams = {}
    for gate1 in ga.gate_set.gates:
        if type(gate1) == Identity:
            continue

        for gate2 in ga.gate_set.gates:
            if type(gate2) == Identity:
                continue

            bigram = gate1.name + "_" + gate2.name
            bigrams[bigram] = []

    for chromosome in population:
        chromosome_bigrams = set()

        for i in range(len(chromosome) - 1):
            gate = chromosome[i]
            successor_gate = chromosome[i + 1]

            if type(gate) == Identity or type(successor_gate) == Identity:
                continue

            bigram = gate.name + "_" + successor_gate.name
            chromosome_bigrams.add(bigram)

        for bigram in bigrams:
            if bigram in chromosome_bigrams:
                bigrams[bigram].append(1)
            else:
                bigrams[bigram].append(0)

    print(f"\nRelevant correlations after generation {generation}:")
    for bigram in bigrams:
        if np.std(bigrams[bigram]) == 0:
            if bigrams[bigram][0] == 0:
                # not present in any chromosome
                pass
            else:
                print(f"\t{bigram}: NAN (present in every chromosome)")
        elif sum(bigrams[bigram]) < ga.params.population_size * 0.1:
            continue
        else:
            correlation = np.corrcoef(bigrams[bigram], fitness_values)[0, 1]

            if correlation > 0.2:
                print(f"\t{bigram}: {correlation}")


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
            OracleConstructor(oracle_circuits),
            HadamardLayer,
            XLayer,
            YLayer,
            ZLayer,
        ],
        qubit_num=3,
    )

    ga_params = GAParams(
        population_size=500,
        generations=20,
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
    optimizer: Optimizer = RemoveRedundanciesOptimizer(
        target_distributions, params=optimizer_params
    )

    genetic_algorithm = GA(gate_set, fitness, optimizer, params=ga_params)

    average_fitness_values = []

    def log_fitness_callback(
        ga: GA,
        population: List[List[Gate]],
        fitness_values: List[float],
        generation: int,
    ) -> None:
        fitness_values = [
            value % 100 for value in fitness_values
        ]  # Remove punishment terms
        average_fitness = mean(fitness_values)
        average_fitness_values.append(average_fitness)

    genetic_algorithm.on_after_generation(compute_bigram_correlations)
    genetic_algorithm.on_after_generation(log_fitness_callback)
    genetic_algorithm.run()

    plt.plot(average_fitness_values)
    plt.xlabel("generation")
    plt.ylabel("mean fitness")
    plt.savefig("tmp/mean_fitness.png")
    plt.show()

    TOP_N = 3
    for chromosome, fitness_value in genetic_algorithm.get_best_chromosomes(n=TOP_N):
        circuit = build_circuit(chromosome, qubit_num=3)

        print(f"\nFitness value: {fitness_value}")
        print(circuit)

        # circuit.draw("mpl")
        # plt.show()
