#!/usr/bin/env python3

from deap import creator, base, tools
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer
import random
from scipy.spatial import distance
from statistics import mean
from typing import Any, List, Callable

from gates import Gate, GateSet, Hadamard, CNot, Identity


POPULATION_SIZE = 20
MAX_GENERATIONS = 10
CROSSOVER_PROBABILITY = 0.5
MUTATION_PROBABILITY = 0.3
CHROMOSOME_LENGTH = 5
QUBIT_NUM = 2


def create_individual(container: Callable, gate_set: GateSet, chromosome_length: int):
    x = []

    for _ in range(chromosome_length):
        current_gate = gate_set.random_gate()
        x.append(current_gate)

    return container(x)


def evaluate_individual(
    chromosome: List[Gate], qubit_num: int, target_distribution: List[float]
):
    circuit = build_circuit(chromosome, qubit_num)
    state_distribution = run_circuit(circuit)

    assert len(state_distribution) == len(target_distribution)

    error = distance.jensenshannon(state_distribution, target_distribution)
    return (error,)


def compute_error(
    state_distribution: List[float], target_distribution: List[float]
) -> float:
    error = 0
    for val, target in zip(state_distribution, target_distribution):
        error += abs(val - target)

    error = error / len(state_distribution)
    return error


def run_circuit(circuit: QuantumCircuit) -> List[float]:
    backend = Aer.get_backend("statevector_simulator")
    job = backend.run(circuit)
    result = job.result()

    output_state = result.get_statevector(circuit, decimals=3)
    state_distribution = output_state.probabilities().tolist()
    return state_distribution


def build_circuit(chromosome: List[Gate], qubit_num: int) -> QuantumCircuit:
    circuit = QuantumCircuit(qubit_num)

    for gate in chromosome:
        circuit = gate.apply_to(circuit)

    return circuit


def swap_gate_mutation(chromosome: List[Gate], gate_set: GateSet, indpb: float):
    for i in range(len(chromosome)):
        if random.random() > indpb:
            continue

        current_gate = gate_set.random_gate()
        chromosome[i] = current_gate

    return (chromosome,)


def operand_mutation(chromosome: List[Gate], indpb: float):
    for i in range(len(chromosome)):
        if random.random() > indpb:
            continue

        current_gate = chromosome[i]
        current_gate.mutate_operands()

    return (chromosome,)


def init_toolbox(gate_set: GateSet, target_distribution: List[float]) -> Any:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        create_individual,
        creator.Individual,
        gate_set=gate_set,
        chromosome_length=CHROMOSOME_LENGTH,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate",
        evaluate_individual,
        qubit_num=QUBIT_NUM,
        target_distribution=target_distribution,
    )
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("gate_mutate", swap_gate_mutation, gate_set=gate_set, indpb=0.3)
    toolbox.register("operand_mutate", operand_mutation, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


if __name__ == "__main__":
    gate_set: GateSet = GateSet(gates=[Hadamard, CNot, Identity], qubit_num=QUBIT_NUM)
    target_distribution: List[float] = [0.5, 0, 0, 0.5]

    toolbox = init_toolbox(gate_set, target_distribution)

    population = toolbox.population(n=POPULATION_SIZE)

    for generation in range(1, MAX_GENERATIONS + 1):
        offspring = [toolbox.clone(ind) for ind in population]

        for i in range(1, len(offspring), 2):
            if random.random() < CROSSOVER_PROBABILITY:
                offspring[i - 1], offspring[i] = toolbox.mate(
                    offspring[i - 1], offspring[i]
                )
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < MUTATION_PROBABILITY:
                (offspring[i],) = toolbox.gate_mutate(offspring[i])
                del offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < MUTATION_PROBABILITY:
                (offspring[i],) = toolbox.operand_mutate(offspring[i])
                del offspring[i].fitness.values

        fitnesses = list(map(toolbox.evaluate, offspring))

        average_fitness = mean([fit for (fit,) in fitnesses])
        print(
            f"Average population fitness at generation {generation}: {average_fitness}"
        )

        for fit, ind in zip(fitnesses, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))

    fitness_values = [ind.fitness.values[0] for ind in offspring]
    best_performer_index = fitness_values.index(min(fitness_values))
    best_performer = offspring[best_performer_index]
    circuit = build_circuit(best_performer, qubit_num=QUBIT_NUM)
    circuit.draw("mpl")
    plt.show()
