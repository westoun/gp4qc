#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Tuple

from fitness import Fitness
from gates import Gate
from .params import OptimizerParams

class Optimizer(ABC):
    """
    """

    def __init__(
        self, target_distributions: List[List[float]], params: OptimizerParams
    ) -> None:
        self.target_distributions = target_distributions
        self.params = params

    @abstractmethod
    def optimize(self, chromosome: List[Gate], fitness: Fitness) -> Tuple[List[Gate], float]:
        ...
