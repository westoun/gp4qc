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
        self, params: FitnessParams
    ) -> None:
        ...

    @abstractmethod
    def evaluate(self, state_distributions: List[List[float]], target_distributions: List[List[float]], 
                 chromosome: List[Gate]) -> float:
        ...
