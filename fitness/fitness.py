#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List

from gates import Gate


class Fitness(ABC):
    """Compute the fitness function of a chromosome including all
    this may entail (constructing the circuit, optimizing it, ...).
    """

    @abstractmethod
    def __init__(self, target_distributions: List[List[float]]) -> None:
        ...

    @abstractmethod
    def evaluate(self, chromosome: List[Gate]) -> float:
        ...
