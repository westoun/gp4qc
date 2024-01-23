#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Tuple

from gates import Gate
from .params import FitnessParams


class Fitness(ABC):
    """Compute the fitness function of a chromosome.
    """

    @abstractmethod
    def __init__(
        self, target_distributions: List[List[float]], params: FitnessParams
    ) -> None:
        ...

    @abstractmethod
    def evaluate(self, chromosome: List[Gate]) -> Tuple[List[Gate], float]:
        ...
