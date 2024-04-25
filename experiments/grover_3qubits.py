#!/usr/bin/env python3

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from statistics import mean
from typing import List, Callable, Type, Union
from uuid import uuid4

from gates import (
    Gate,
    GateSet,
    H,
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
    HLayer,
    XLayer,
    YLayer,
    ZLayer,
    CombinedGate,
    CombinedGateConstructor,
    RY,
    RX,
    RZ,
    CRY,
    CRZ,
    CRX,
    Phase,
    SwapLayer,
    CH,
)
from ga import GA, GAParams
from fitness import (
    Fitness,
    Jensensshannon,
    FitnessParams,
    SpectorFitness,
    BaselineFitness,
    IndirectQAFitness,
    DirectQAFitness,
)
from fitness.validity_checks import uses_oracle, uses_hadamard_layer
from optimizer import (
    Optimizer,
    DoNothingOptimizer,
    OptimizerParams,
    build_circuit,
    RemoveRedundanciesOptimizer,
    NumericalOptimizer,
)
from utils.logging import (
    log_experiment_details,
    log_fitness,
    log_event,
    GATE_ADDED_EVENT,
    ALGORITHM_RESTART_EVENT,
)
from gates.utils import extract_ngram_types, get_unique_chomosomes, construct_ngram_name
from utils.formatting import state_to_distribution

# Place experiment id creation outside of main function
# to avoid having to pass it through multiple layer of
# nested function calls.
EXPERIMENT_ID = f"grover_3qubits_{uuid4()}"

# Describe in which configuration the experiment is being
# run. Especially: which treatments are being applied?
DESCRIPTION = ""


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


def compute_bigram_correlations(
    ga: GA,
    population: List[List[Gate]],
    fitness_values: List[float],
    generation: int,
) -> None:
    population = [ga.toolbox.clone(ind) for ind in population]

    # Get unique chromosomes and recompute fitness values to avoid
    # distorted correlation computation due to high amount of
    # duplicates in the population.
    unique_chromosomes = get_unique_chomosomes(population)
    fitness_values = [chromosome.fitness.values[0] for chromosome in unique_chromosomes]

    bigrams = {}
    bigram_types = {}

    # Note: in the gate_set we have constructor classes, while
    # the chromosomes themselves contain the constructed gates.
    # Since combined gates require the constructor classes,
    # bigram_types are collected here. The construct_ngram_name
    # makes sure that both classes and instances are mapped to
    # the same name space.
    for gate1 in ga.gate_set.gates:
        if type(gate1) == Identity:
            continue

        for gate2 in ga.gate_set.gates:
            if type(gate2) == Identity:
                continue

            bigram = construct_ngram_name([gate1, gate2])

            bigrams[bigram] = []
            bigram_types[bigram] = extract_ngram_types([gate1, gate2])

    for chromosome in unique_chromosomes:
        chromosome_bigrams = set()

        for i in range(len(chromosome) - 1):
            gate = chromosome[i]
            successor_gate = chromosome[i + 1]

            if type(gate) == Identity or type(successor_gate) == Identity:
                continue

            bigram = construct_ngram_name([gate, successor_gate])
            chromosome_bigrams.add(bigram)

        for bigram in bigrams:
            if bigram in chromosome_bigrams:
                bigrams[bigram].append(1)
            else:
                bigrams[bigram].append(0)

    for bigram in bigrams:

        # Check for arbitrary support level
        if sum(bigrams[bigram]) < len(unique_chromosomes) * 0.05:
            continue

        elif np.std(bigrams[bigram]) == 0:
            if bigrams[bigram][0] == 0:
                # not present in any chromosome
                pass
            else:

                NewCombinedGate = CombinedGateConstructor(bigram_types[bigram])

                if not ga.gate_set.contains(NewCombinedGate):
                    ga.gate_set.append(NewCombinedGate)

                    log_event(
                        experiment_id=EXPERIMENT_ID,
                        event_type=GATE_ADDED_EVENT,
                        payload=f"Creating {bigram} as separate gate due to presence in every chromosome.",
                        target_path="results/events.csv",
                    )

        else:
            correlation = np.corrcoef(bigrams[bigram], fitness_values)[0, 1]

            # Look for negative correlation, since lower fitness values are better
            if correlation < -0.25:

                NewCombinedGate = CombinedGateConstructor(bigram_types[bigram])

                if not ga.gate_set.contains(NewCombinedGate):
                    ga.gate_set.append(NewCombinedGate)

                    log_event(
                        experiment_id=EXPERIMENT_ID,
                        event_type=GATE_ADDED_EVENT,
                        payload=f"Creating {bigram} as separate gate due to fitness correlation of {correlation}.",
                        target_path="results/events.csv",
                    )


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
            H,
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
            HLayer,
            XLayer,
            YLayer,
            ZLayer,
            SwapLayer,
            RY,
            RX,
            RZ,
            CRY,
            CRZ,
            CRX,
            Phase,
            CH,
        ],
        qubit_num=3,
    )

    ga_params = GAParams(
        population_size=1000,
        generations=800,
        crossover_prob=0.4,
        swap_gate_mutation_prob=0.03,
        swap_order_mutation_prob=0,
        operand_mutation_prob=0,
        chromosome_length=30,
        log_average_fitness=False,
        log_average_fitness_at=1,
        cpu_count=28,
        elitism_percentage=0.08,
    )

    fitness_params = FitnessParams(validity_checks=[], classical_oracle_count=2**3)
    # fitness: Fitness = SpectorFitness(params=fitness_params)
    fitness: Fitness = BaselineFitness(params=fitness_params)
    # fitness: Fitness = IndirectQAFitness(params=fitness_params)
    # fitness: Fitness = DirectQAFitness(params=fitness_params)

    optimizer_params = OptimizerParams(qubit_num=3, measurement_qubit_num=3, max_iter=8)
    optimizer: Optimizer = NumericalOptimizer(
        target_distributions, params=optimizer_params
    )

    genetic_algorithm = GA(gate_set, fitness, optimizer, params=ga_params)

    log_experiment_details(
        ga=genetic_algorithm,
        experiment_id=EXPERIMENT_ID,
        target_path="results/experiments.csv",
        description=DESCRIPTION,
    )

    mean_fitness_values = []

    def log_fitness_callback(
        ga: GA,
        population: List[List[Gate]],
        fitness_values: List[float],
        generation: int,
    ) -> None:
        best_chromosome, best_fitness_value = ga.get_best_chromosomes(1)[0]
        mean_fitness_value = mean(fitness_values)

        mean_fitness_values.append(mean_fitness_value)

        log_fitness(
            experiment_id=EXPERIMENT_ID,
            generation=generation,
            best_fitness_value=best_fitness_value,
            mean_fitness_value=mean_fitness_value,
            best_chromosome=best_chromosome,
            target_path="results/fitness_values.csv",
        )

    # genetic_algorithm.on_after_generation(compute_bigram_correlations)
    genetic_algorithm.on_after_generation(log_fitness_callback)

    genetic_algorithm.run()

    plt.plot(mean_fitness_values)
    plt.xlabel("generation")
    plt.ylabel("mean fitness")
    plt.savefig(f"results/{EXPERIMENT_ID}_mean_fitness.png")
    # plt.show()

    TOP_N = 3
    for chromosome, fitness_value in genetic_algorithm.get_best_chromosomes(n=TOP_N):
        circuit = build_circuit(chromosome, qubit_num=3)

        print(f"\nFitness value: {fitness_value}")
        print(circuit)

        # circuit.draw("mpl")
        # plt.show()
