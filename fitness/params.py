#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class FitnessParams:
    qubit_num: int
    measurement_qubit_num: int
