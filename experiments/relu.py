#!/usr/bin/env python3

from math import pi
from random import uniform
from typing import List

from gates import (
    Gate,
    GateSet,
    H,
    X,
    Y,
    Z,
    RX,
    RZ,
    RY,
    Identity,
    Phase,
    PhaseEncoding,
    RXEncoding,
    RYEncoding,
    RZEncoding,
    InputEncodingConstructor,
    CX,
    CY,
    CZ,
    CRY,
    CRZ,
    CRX,
    Swap,
)
from ga import GA, GAParams
from fitness import Fitness, Jensensshannon, FitnessParams
from optimizer import (
    Optimizer,
    NumericalOptimizer,
    DoNothingOptimizer,
    OptimizerParams,
    build_circuit,
    run_circuit,
)


def run_construct_relu():
    input_values: List[List[float]] = []
    target_distributions: List[List[float]] = []

    CASE_COUNT = 30
    for _ in range(CASE_COUNT):
        input_value = uniform(-1 * pi, pi)
        input_values.append([input_value])

        if input_value < 0:
            target_distributions.append([1, 0])
        else:
            target_distributions.append([1 - input_value / pi, input_value / pi])

    gate_set: GateSet = GateSet(
        gates=[
            H,
            X,
            Y,
            Z,
            RX,
            RZ,
            RY,
            Phase,
            Identity,
            CX,
            CY,
            CZ,
            CRY,
            CRZ,
            CRX,
            Swap,
            InputEncodingConstructor(input_values, PhaseEncoding),
            InputEncodingConstructor(input_values, RXEncoding),
            InputEncodingConstructor(input_values, RYEncoding),
            InputEncodingConstructor(input_values, RZEncoding),
        ],
        qubit_num=2,
    )
    ga_params = GAParams(
        population_size=100,
        generations=10,
        crossover_prob=0.4,
        swap_gate_mutation_prob=0.15,
        chromosome_length=5,
        log_average_fitness_at=1,
        fitness_threshold=0.001,
        fitness_threshold_at=1,
    )

    fitness_params = FitnessParams()
    fitness: Fitness = Jensensshannon(params=fitness_params)

    optimizer_params = OptimizerParams(
        qubit_num=2, measurement_qubit_num=1, max_iter=20
    )
    optimizer: Optimizer = NumericalOptimizer(
        target_distributions, params=optimizer_params
    )

    genetic_algorithm = GA(gate_set, fitness, optimizer, params=ga_params)

    genetic_algorithm.run()

    TOP_N = 3
    for chromosome, fitness_value in genetic_algorithm.get_best_chromosomes(n=TOP_N):
        print("")
        print(chromosome)
        print(fitness_value)
