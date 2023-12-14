#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List

from gates import Gate


class Fitness(ABC):
    @abstractmethod
    def __init__(self, target_distributions: List[List[float]]) -> None:
        ...

    @abstractmethod
    def evaluate(self, chromosome: List[Gate]) -> float:
        ...
