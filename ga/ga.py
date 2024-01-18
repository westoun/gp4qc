#!/usr/bin/env python3

from multiprocessing import Pool
import random
from statistics import mean
from typing import Any, List, Tuple
import warnings

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
        # Catch reinitialization warning if multiple GAs are used
        # in one run.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
                    offspring[i] = toolbox.gate_mutate(offspring[i])
                    del offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < self.params.operand_mutation_prob:
                    offspring[i] = toolbox.operand_mutate(offspring[i])
                    del offspring[i].fitness.values


            # If no amount of workers is specified, os.cpu_count() is used.
            with Pool() as pool: 
                offspring = pool.map(toolbox.evaluate, offspring)

            population = toolbox.select(offspring, k=len(population))
            self._evolved_population = population

            if (
                self.params.log_average_fitness
                and generation % self.params.log_average_fitness_at == 0
            ):
                average_fitness = mean([ind.fitness.values[0] for ind in offspring])
                print(
                    f"Average population fitness at generation {generation}: {average_fitness}"
                )

            _, fitness_at = self.get_best_chromosomes(
                n=self.params.fitness_threshold_at + 1
            )[self.params.fitness_threshold_at]

            if fitness_at <= self.params.fitness_threshold:
                if self.params.log_average_fitness:
                    print(
                        "\tFound good enough solution. Skipping remaining generations."
                    )

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
