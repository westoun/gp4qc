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

# TODO: move to utils folder
from .grover_3qubits import compute_bigram_correlations

# Place experiment id creation outside of main function
# to avoid having to pass it through multiple layer of
# nested function calls.
EXPERIMENT_ID = f"bernstein_vazirani_3qubits_{uuid4()}"

# Describe in which configuration the experiment is being
# run. Especially: which treatments are being applied?
DESCRIPTION = ""


def construct_oracle_circuit(target_state: List[int]) -> QuantumCircuit:
    circuit = QuantumCircuit(len(target_state) + 1)

    for i, qubit_state in enumerate(target_state):
        if qubit_state == 1:
            # Set ancilla as target qubit
            circuit.cx(control_qubit=i, target_qubit=len(target_state))

    return circuit


def run_bernstein_vazirani():
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
        qubit_num=4,
    )

    ga_params = GAParams(
        population_size=1000,
        generations=500,
        crossover_prob=0.4,
        swap_gate_mutation_prob=0.03,
        swap_order_mutation_prob=0,
        operand_mutation_prob=0,
        chromosome_length=15,
        log_average_fitness=True,
        log_average_fitness_at=1,
        cpu_count=25,
        elitism_percentage=0.1,
    )

    fitness_params = FitnessParams(validity_checks=[], classical_oracle_count=2**3)
    fitness: Fitness = BaselineFitness(params=fitness_params)
    # fitness: Fitness = IndirectQAFitness(params=fitness_params)
    # fitness: Fitness = DirectQAFitness(params=fitness_params)

    optimizer_params = OptimizerParams(qubit_num=4, measurement_qubit_num=3, max_iter=8)
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
        circuit = build_circuit(chromosome, qubit_num=4)

        print(f"\nFitness value: {fitness_value}")
        print(circuit)

        # circuit.draw("mpl")
        # plt.show()
