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
    CombinedGate,
    CombinedGateConstructor,
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
from utils.logging import log_experiment_details, log_fitness


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


def construct_ngram_name(gates: List[Gate]) -> str:
    gate_names = []
    for gate in gates:
        if type(gate) in [CombinedGate, CombinedGateConstructor]:
            for GateType in gate.GateTypes:
                if GateType == Oracle or type(GateType) == OracleConstructor:
                    gate_names.append("oracle")
                else:
                    gate_names.append(GateType.name)
        elif type(gate) in [Oracle, OracleConstructor]:
            # TA: How to handle multiple oracles?
            gate_names.append("oracle")
        else:
            gate_names.append(gate.name)
    return "_".join(gate_names)


def extract_ngram_types(
    gates: List[Union[Type, CombinedGateConstructor, OracleConstructor]]
) -> List[Type]:
    gate_types = []
    for gate in gates:
        if type(gate) == CombinedGateConstructor:
            gate_types.extend(gate.GateTypes)
        elif type(gate) == OracleConstructor:
            gate_types.append(gate)
        elif type(gate) in [Oracle, CombinedGate]:
            raise ValueError("Missing information! Should be constructor classes!")
        else:
            gate_types.append(gate)
    return gate_types


def compute_bigram_correlations(
    ga: GA,
    population: List[List[Gate]],
    fitness_values: List[float],
    generation: int,
) -> None:
    if generation < 10:
        return

    population = [ga.toolbox.clone(ind) for ind in population]

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

    for chromosome in population:
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
        if sum(bigrams[bigram]) < ga.params.population_size * 0.1:
            continue

        elif np.std(bigrams[bigram]) == 0:
            if bigrams[bigram][0] == 0:
                # not present in any chromosome
                pass
            else:
                print(
                    f"\tCorrelation of {bigram} at generation {generation}: NAN (present in every chromosome)"
                )

                ga.stop()

                NewCombinedGate = CombinedGateConstructor(bigram_types[bigram])
                ga.gate_set.append(NewCombinedGate)

        else:
            correlation = np.corrcoef(bigrams[bigram], fitness_values)[0, 1]

            if correlation > 0.2:
                print(
                    f"\tCorrelation of {bigram} at generation {generation}: {correlation}"
                )

                ga.stop()

                NewCombinedGate = CombinedGateConstructor(bigram_types[bigram])
                ga.gate_set.append(NewCombinedGate)


def run_grover():
    EXPERIMENT_ID = f"grover_3qubits_{uuid4()}"

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
        population_size=100,  # 700,
        generations=10,
        crossover_prob=0.4,
        swap_gate_mutation_prob=0.1,
        swap_order_mutation_prob=0.1,
        operand_mutation_prob=0.1,
        chromosome_length=10,
        log_average_fitness_at=-1,
    )

    fitness_params = FitnessParams(validity_checks=[])
    fitness: Fitness = SpectorFitness(params=fitness_params)

    optimizer_params = OptimizerParams(qubit_num=3, measurement_qubit_num=3)
    optimizer: Optimizer = RemoveRedundanciesOptimizer(
        target_distributions, params=optimizer_params
    )

    genetic_algorithm = GA(gate_set, fitness, optimizer, params=ga_params)

    log_experiment_details(
        ga=genetic_algorithm,
        experiment_id=EXPERIMENT_ID,
        target_path="results/experiments.csv",
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
    # for i in range(5):
    #     genetic_algorithm.run()

    #     if not genetic_algorithm.has_been_stopped():
    #         break

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
