#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import List, Callable

from gates import Gate


@dataclass
class FitnessParams:
    qubit_num: int
    measurement_qubit_num: int
    # validity checks identify invalid chromosomes.
    # based on these checks, fitness functions can decrease
    # the likelyhood of invalid chromosomes to pass on to the
    # next generation.
    validity_checks: List[Callable[[List[Gate]], bool]] = field(
        default_factory=lambda: []
    )
