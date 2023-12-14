#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class GAParams:
    population_size: int
    generations: int
    crossover_prob: float
    swap_mutation_prob: float
    operand_mutation_prob: float
    chromosome_length: int
    fitness_threshold: float = 0
    fitness_threshold_at: int = 0