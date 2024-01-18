#!/usr/bin/env python3

from deap import creator, base, tools
import random
from typing import Any, List, Callable, Tuple

from gates import Gate, GateSet
from fitness import Fitness


def create_individual(container: Callable, gate_set: GateSet, chromosome_length: int):
    x = []

    for _ in range(chromosome_length):
        current_gate = gate_set.random_gate()
        x.append(current_gate)

    return container(x)


def swap_gate_mutation(
    chromosome: List[Gate], gate_set: GateSet, indpb: float
) -> List[Gate]:
    for i in range(len(chromosome)):
        if random.random() > indpb:
            continue

        current_gate = gate_set.random_gate()
        chromosome[i] = current_gate

    return chromosome


def operand_mutation(chromosome: List[Gate], indpb: float) -> List[Gate]:
    for i in range(len(chromosome)):
        if random.random() > indpb:
            continue

        current_gate = chromosome[i]
        current_gate.mutate_operands()

    return chromosome


def evaluate_individual(chromosome: List[Gate], fitness: Fitness) -> List[Gate]:
    fitness_score = fitness.evaluate(chromosome)
    chromosome.fitness.values = fitness_score
    return chromosome


def init_toolbox(gate_set: GateSet, chromosome_length: int, fitness: Fitness) -> Any:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        create_individual,
        creator.Individual,
        gate_set=gate_set,
        chromosome_length=chromosome_length,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate",
        evaluate_individual,
        fitness=fitness,
    )
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("gate_mutate", swap_gate_mutation, gate_set=gate_set, indpb=0.1)
    toolbox.register("operand_mutate", operand_mutation, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox
