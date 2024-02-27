import numpy as np 
from typing import List

from gates import (
    Gate,
    GateSet,
    Hadamard,
    X,
    Y,
    Z,
    CX,
    Swap,
    CY,
    CZ,
    CCX,
    Identity,
    InputEncoding,
    BinaryEncoding,
    CombinedGate
)

from ga import GA, GAParams
from fitness import Fitness, Jensensshannon, FitnessParams
from fitness.validity_checks import has_exactly_1_input
from optimizer import Optimizer, DoNothingOptimizer, OptimizerParams, build_circuit


def run_half_adder():
    input_values: List[List[int]] = [[0, 0], [0, 1], [1, 0], [1, 1]]
    target_distributions: List[List[float]] = [
        np.kron([1, 0], [1, 0]),# 00 => 00
        np.kron([1, 0], [0, 1]),# 01 => 01
        np.kron([1, 0], [0, 1]),# 10 => 01
        np.kron([0, 1], [1, 0]),# 11 => 10
    ]

    gate_set: GateSet = GateSet(
        gates=[
            Hadamard,
            Y,
            Z,
            CY,
            CZ,
            Swap,
            X,
            CX,
            CCX,
            Identity,
            BinaryEncoding.init_circuits(input_values, qubit_num=2),
        ],
        qubit_num=3,
    )

    ga_params = GAParams(
        population_size=200,
        generations=40,
        crossover_prob=0.5,
        swap_gate_mutation_prob=0.1,
        swap_order_mutation_prob=0.1,
        operand_mutation_prob=0.1,
        chromosome_length=5 + 1,  # + 1 for input layer
        fitness_threshold=0.1,
        log_average_fitness_at=1,
    )

    fitness_params = FitnessParams(
        validity_checks=[has_exactly_1_input]
    )
    fitness: Fitness = Jensensshannon(
        params=fitness_params
    )

    optimizer_params = OptimizerParams(qubit_num=3, measurement_qubit_num=2)
    optimizer: Optimizer = DoNothingOptimizer(target_distributions, params = optimizer_params)


    genetic_algorithm = GA(gate_set, fitness, optimizer, params=ga_params)
    genetic_algorithm.run()

    TOP_N = 3
    for chromosome, fitness_value in genetic_algorithm.get_best_chromosomes(n=TOP_N):
        circuit = build_circuit(chromosome, qubit_num=3)

        print(f"\nFitness value: {fitness_value}")
        print(circuit)

        # circuit.draw("mpl")
        # plt.show()
