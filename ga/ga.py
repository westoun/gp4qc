#!/usr/bin/env python3

from math import floor
from multiprocessing import Pool
import os
import random
from statistics import mean
from typing import Any, List, Tuple, Callable
import warnings

from .params import GAParams, default_params
from gates import Gate, GateSet
from .utils import init_toolbox
from fitness import Fitness
from optimizer import Optimizer


class GA:
    """Wrapper class for the genetic algorithm code."""

    gate_set: GateSet

    evolved_population: List[Gate]
    _after_generation_callbacks: List[Callable]
    _on_completion_callbacks: List[Callable]

    _stopped: bool

    def __init__(
        self,
        gate_set: GateSet,
        fitness: Fitness,
        optimizer: Optimizer,
        params: GAParams = default_params,
    ) -> None:
        self.evolved_population = []
        self._after_generation_callbacks = []
        self._on_completion_callbacks = []
        self._stopped = False

        self.gate_set = gate_set
        self.fitness = fitness
        self.optimizer = optimizer
        self.params = params

    def on_after_generation(self, callback: Callable) -> None:
        self._after_generation_callbacks.append(callback)

    def on_completion(self, callback: Callable) -> None:
        self._on_completion_callbacks.append(callback)

    def __call__(self):
        return self.run()

    def run(self):
        self._stopped = False

        # Catch reinitialization warning if multiple GAs are used
        # in one run.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.toolbox = toolbox = init_toolbox(
                self.gate_set,
                self.params.chromosome_length,
                self.fitness,
                self.optimizer,
            )

        population = toolbox.population(n=self.params.population_size)

        for generation in range(1, self.params.generations + 1):
            if self._stopped:
                break

            if generation == 1:
                elite = []
            else:
                elite = toolbox.select_best(population, k=self.params.elitism_count)
                # Create deep copy to avoid adjusting elite chromosomes through mutation
                elite = [toolbox.clone(ind) for ind in elite]

            offspring = [toolbox.clone(ind) for ind in population]
            random.shuffle(offspring)

            for i in range(1, len(offspring), 2):
                if random.random() < self.params.crossover_prob:
                    offspring[i - 1], offspring[i] = toolbox.mate(
                        offspring[i - 1], offspring[i]
                    )

            if self.params.swap_gate_mutation_prob > 0:
                for i in range(len(offspring)):
                    for j in range(len(offspring[i])):
                        if random.random() < self.params.swap_gate_mutation_prob:
                            offspring[i] = toolbox.swap_gate_mutate(
                                offspring[i], gate_idx=j
                            )

            if self.params.operand_mutation_prob > 0:
                for i in range(len(offspring)):
                    for j in range(len(offspring[i])):
                        if random.random() < self.params.operand_mutation_prob:
                            offspring[i] = toolbox.operand_mutate(
                                offspring[i], gate_idx=j
                            )

            if self.params.swap_order_mutation_prob > 0:
                for i in range(len(offspring)):
                    for j in range(len(offspring[i])):
                        if random.random() < self.params.swap_order_mutation_prob:
                            offspring[i] = toolbox.swap_order_mutate(
                                offspring[i], gate1_idx=j
                            )

            with Pool(processes=self.params.cpu_count) as pool:
                offspring = pool.map(toolbox.evaluate, offspring)

            population = elite + toolbox.select(
                offspring, k=self.params.population_size - len(elite)
            )
            self.evolved_population = population

            if (
                self.params.log_average_fitness
                and generation % self.params.log_average_fitness_at == 0
            ):
                average_fitness = mean([ind.fitness.values[0] for ind in population])
                print(
                    f"Average population fitness at generation {generation}: {average_fitness}"
                )

            # Call callbacks
            fitness_values = [chromosome.fitness.values[0] for chromosome in population]
            for callback in self._after_generation_callbacks:
                callback(self, population, fitness_values, generation)

            # Check early abort condition (fitness value at.)
            _, fitness_at = self.get_best_chromosomes(
                n=self.params.fitness_threshold_at + 1
            )[self.params.fitness_threshold_at]

            if fitness_at <= self.params.fitness_threshold:
                if self.params.log_average_fitness:
                    print(
                        "\tFound good enough solution. Skipping remaining generations."
                    )

                break

        fitness_values = [
            chromosome.fitness.values[0] for chromosome in self.evolved_population
        ]
        for callback in self._on_completion_callbacks:
            callback(self, population, fitness_values, generation)

    def get_best_chromosomes(self, n: int = 1) -> List[Tuple[List[Gate], float]]:
        assert self.evolved_population is not None

        self.evolved_population.sort(key=lambda item: item.fitness.values[0])
        top_n_chromosomes = self.evolved_population[:n]

        result = [
            (chromosome, chromosome.fitness.values[0])
            for chromosome in top_n_chromosomes
        ]
        return result

    def stop(self) -> None:
        self._stopped = True

    def has_been_stopped(self) -> bool:
        return self._stopped
