#!/usr/bin/env python3

from dataclasses import dataclass, field

@dataclass
class OptimizerParams:
    qubit_num: int = 2
    measurement_qubit_num: int = 2
    tolerance: float = 0
    max_iter: int = 10

default_params = OptimizerParams()