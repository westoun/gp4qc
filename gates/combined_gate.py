#!/usr/bin/env python3

from qiskit import QuantumCircuit
from typing import List, Union, Tuple, Type

from .gate import Gate
from .multicase_gate import MultiCaseGate
from .optimizable import OptimizableGate


# Needs to overwrite everything but the class methods
class CombinedGate(OptimizableGate, MultiCaseGate):
    gate_name: str = "combined_gate"

    GateTypes: List[Type] = None
    gates: List[Gate] = []

    def __init__(self, qubit_num: int) -> None:
        print("init called")
        assert self.GateTypes is not None, "No GateTypes have been provided."

        # This re-assignment is necessary to make variables
        # set through cls available as instance variables in
        # a multiprocessing setup.
        self.GateTypes = self.GateTypes
        self._qubit_num = qubit_num

        for GateType in self.GateTypes:
            gate = GateType(qubit_num=qubit_num)
            self.gates.append(gate)

    @classmethod
    def set_gate_types(cls, GateTypes: List[Type]) -> None:
        cls.GateTypes = GateTypes

    # General gate functions
    def mutate_operands(self) -> None: 
        for gate in self.gates:
            gate.mutate_operands()

    def apply_to(self, circuit: QuantumCircuit) -> QuantumCircuit: 
        print("gates in combined:", len(self.gates))
        for gate in self.gates:
            circuit = gate.apply_to(circuit)
        return circuit

    def __repr__(self) -> str: 
        combined_repr = f"""{self.gate_name}({', '.join(
            [gate.__repr__() for gate in self.gates]
        )})"""
        return combined_repr

    @property
    def name(self) -> str:
        combined_name = f"""{self.gate_name}[{', '.join(
            [gate.name for gate in self.gates]
        )}]"""
        return combined_name

    @property
    def is_input(self) -> bool:
        for gate in self.gates:
            if gate.is_input:
                return True

        return False

    @property
    def is_optimizable(self) -> bool:
        for gate in self.gates:
            if gate.is_optimizable:
                return True

        return False

    @property
    def is_oracle(self) -> bool:
        for gate in self.gates:
            if gate.is_oracle:
                return True

        return False

    @property
    def is_multicase(self) -> bool:
        for gate in self.gates:
            if gate.is_multicase:
                return True

        return False

    # Optimizable gate functions
    @property
    def params(self) -> List[float]: 
        param_vector = []

        for gate in self.gates:
            if gate.is_optimizable:
                param_vector.extend(gate.params)

        return param_vector

    @property
    def bounds(self) -> List[Union[Tuple[float, float], None]]: 
        bounds_vector = []

        for gate in self.gates:
            if gate.is_optimizable:
                bounds_vector.extend(gate.bounds)

        return bounds_vector

    @property
    def set_params(self, params: List[float]) -> None: 
        for gate in self.gates:
            if gate.is_optimizable:
                gate_params, params = (
                    params[: gate.param_count],
                    params[gate.param_count :],
                )
                gate.set_params(gate_params)

    # Multicase gate functions
    def set_case_index(self, index: int) -> MultiCaseGate:
        for gate in self.gates:
            if gate.is_multicase:
                gate.set_case_index(index)
