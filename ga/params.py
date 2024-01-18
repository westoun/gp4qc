#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class GAParams:
    population_size: int
    generations: int
    crossover_prob: float
    swap_gate_mutation_prob: float
    chromosome_length: int
    operand_mutation_prob: float = 0
    swap_order_mutation_prob: float = 0 
    fitness_threshold: float = 0
    fitness_threshold_at: int = 0
    log_average_fitness: bool = True 
    log_average_fitness_at: int = 5