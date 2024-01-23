#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Tuple

from fitness import Fitness
from gates import Gate

class Optimizer(Fitness, ABC):
    """Wrapper class around Fitness, used to optimize a 
    circuit based on the specified fitness function class.
    """

    def __init__(
        self, fitness: Fitness
    ) -> None:
        self.fitness = fitness 

    @abstractmethod
    def evaluate(self, chromosome: List[Gate]) -> Tuple[List[Gate], float]:
        ...
