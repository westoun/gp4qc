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
)

from ga import GA, GAParams
from fitness import Fitness, Jensensshannon, build_circuit, MatchCount


def run_half_adder():
    QUBIT_NUM = 3
    MEASUREMENT_QUBIT_NUM = 2

    input_values: List[List[int]] = [[0, 0], [0, 1], [1, 0], [1, 1]]
    target_distributions: List[List[float]] = [
        [1, 0, 0, 0],  # 00 => 00
        [0, 1, 0, 0],  # 01 => 01
        [0, 1, 0, 0],  # 10 => 01
        [0, 0, 1, 0],  # 11 => 10
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
            BinaryEncoding.init_circuits(input_values, qubit_num=QUBIT_NUM),
        ],
        qubit_num=QUBIT_NUM,
    )

    ga_params = GAParams(
        population_size=200,
        generations=40,
        crossover_prob=0.5,
        swap_mutation_prob=0.1,
        operand_mutation_prob=0.1,
        chromosome_length=5 + 1,  # + 1 for input layer
        fitness_threshold=0.1,
        fitness_threshold_at=3,
        log_average_fitness_at=1,
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
