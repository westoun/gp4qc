#!/usr/bin/env python3

from dataclasses import dataclass
from math import floor

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
    elitism_percentage: float = 0

    @property
    def elitism_count(self) -> int:
        return floor(self.elitism_percentage * self.population_size)