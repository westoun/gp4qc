#!/usr/bin/env python3

import random
from statistics import mean
from typing import Any, List, Tuple

from .params import GAParams
from gates import Gate, GateSet
from .utils import init_toolbox
from fitness import Fitness


class GA:
    _evolved_population: List[Gate] = None

    def __init__(
        self, gate_set: GateSet, fitness: Fitness, qubit_num: int, params: GAParams
    ) -> None:
        self.gate_set = gate_set
        self.fitness = fitness
        self.qubit_num = qubit_num
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

    def get_best_chromosomes(self, n: int = 1) -> List[Tuple[List[Gate], float]]:
        # TODO: Implement n > 1

        assert self._evolved_population is not None

        fitness_values = [ind.fitness.values[0] for ind in self._evolved_population]
        best_performer_index = fitness_values.index(min(fitness_values))
        best_performer = self._evolved_population[best_performer_index]
        fitness_value = fitness_values[best_performer_index]
        
        return [(best_performer, fitness_value)]
