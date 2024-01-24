#!/usr/bin/env python3

from dataclasses import dataclass, field

@dataclass
class OptimizerParams:
    qubit_num: int = 2
    measurement_qubit_num: int = 2

default_params = OptimizerParams()