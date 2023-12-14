#!/usr/bin/env python3

import matplotlib.pyplot as plt
from typing import List

from gates import Gate, GateSet, Hadamard, X, Y, Z, CX, CY, CZ, Identity
from ga import GA, GAParams
from fitness import Fitness, Jensensshannon, build_circuit


def run_bell_state_2qubits():
    QUBIT_NUM = 2

    gate_set: GateSet = GateSet(
        gates=[Hadamard, X, Y, Z, CX, CY, CZ, Identity], qubit_num=QUBIT_NUM
    )
    target_distributions: List[List[float]] = [[0.5, 0, 0, 0.5]]

    ga_params = GAParams(
        population_size=100,
        generations=20,
        crossover_prob=0.5,
        swap_mutation_prob=0.1,
        operand_mutation_prob=0.4,
        chromosome_length=4,
    )

    fitness: Fitness = Jensensshannon(
        target_distributions=target_distributions, qubit_num=QUBIT_NUM
    )

    genetic_algorithm = GA(gate_set, fitness, qubit_num=3, params=ga_params)
    genetic_algorithm.run()

    best_performer, fitness_value = genetic_algorithm.get_best_chromosomes(n=1)[0]
    circuit = build_circuit(best_performer, qubit_num=QUBIT_NUM)

    print(f"\nBest performing fitness value: {fitness_value}")
    print(best_performer)
    print(circuit)

    circuit.draw("mpl")
    plt.show()
