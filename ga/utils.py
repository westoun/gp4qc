#!/usr/bin/env python3

from deap import creator, base, tools
import random
from typing import Any, List, Callable, Tuple

from gates import Gate, GateSet
from fitness import Fitness
from optimizer import Optimizer


def create_individual(container: Callable, gate_set: GateSet, chromosome_length: int):
    x = []

    for _ in range(chromosome_length):
        current_gate = gate_set.random_gate()
        x.append(current_gate)

    return container(x)


def swap_gate_mutation(
    chromosome: List[Gate],
    gate_idx: int,
    gate_set: GateSet,
) -> List[Gate]:
    new_gate = gate_set.random_gate()
    chromosome[gate_idx] = new_gate

    return chromosome


def swap_order_mutation(chromosome: List[Gate], gate1_idx: int) -> List[Gate]:
    gate2_idx = random.randint(0, len(chromosome) - 1)

    chromosome[gate1_idx], chromosome[gate2_idx] = (
        chromosome[gate2_idx],
        chromosome[gate1_idx],
    )
    return chromosome


def operand_mutation(chromosome: List[Gate], gate_idx: int) -> List[Gate]:
    chromosome[gate_idx].mutate_operands()
    return chromosome


def evaluate_individual(
    chromosome: List[Gate], fitness: Fitness, optimizer: Optimizer
) -> List[Gate]:
    chromosome, fitness_score = optimizer.optimize(chromosome, fitness)
    chromosome.fitness.values = (fitness_score,)
    return chromosome


def init_toolbox(
    gate_set: GateSet, chromosome_length: int, fitness: Fitness, optimizer: Optimizer
) -> Any:
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
        "evaluate", evaluate_individual, fitness=fitness, optimizer=optimizer
    )
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("swap_gate_mutate", swap_gate_mutation, gate_set=gate_set)
    toolbox.register("swap_order_mutate", swap_order_mutation)
    toolbox.register("operand_mutate", operand_mutation)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("select_best", tools.selBest)
    return toolbox
