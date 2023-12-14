#!/usr/bin/env python3

import random
from statistics import mean
from typing import Any, List, Tuple

from .params import GAParams
from gates import Gate, GateSet
from .utils import init_toolbox
from fitness import Fitness


class GA:
    """Wrapper class for the genetic algorithm code."""

    _evolved_population: List[Gate] = None

    def __init__(self, gate_set: GateSet, fitness: Fitness, params: GAParams) -> None:
        self.gate_set = gate_set
        self.fitness = fitness
        self.params = params

    def __call__(self):
        return self.run()

    def run(self):
        toolbox = init_toolbox(
            self.gate_set, self.params.chromosome_length, self.fitness
        )

        population = toolbox.population(n=self.params.population_size)

        for generation in range(1, self.params.generations + 1):
            offspring = [toolbox.clone(ind) for ind in population]

            for i in range(1, len(offspring), 2):
                if random.random() < self.params.crossover_prob:
                    offspring[i - 1], offspring[i] = toolbox.mate(
                        offspring[i - 1], offspring[i]
                    )
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < self.params.swap_mutation_prob:
                    (offspring[i],) = toolbox.gate_mutate(offspring[i])
                    del offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < self.params.operand_mutation_prob:
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
            self._evolved_population = population

            _, fitness_at = self.get_best_chromosomes(
                n=self.params.fitness_threshold_at + 1
            )[self.params.fitness_threshold_at]
            if fitness_at <= self.params.fitness_threshold:
                print("Found good enough solution. Aborting GA.")
                return

    def get_best_chromosomes(self, n: int = 1) -> List[Tuple[List[Gate], float]]:
        assert self._evolved_population is not None

        self._evolved_population.sort(key=lambda item: item.fitness.values[0])
        top_n_chromosomes = self._evolved_population[:n]

        result = [
            (chromosome, chromosome.fitness.values[0])
            for chromosome in top_n_chromosomes
        ]
        return result
