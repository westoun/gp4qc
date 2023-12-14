#!/usr/bin/env python3

from qiskit import QuantumCircuit, Aer
from typing import List

from gates import Gate, InputEncoding


def run_circuit(circuit: QuantumCircuit) -> List[float]:
    backend = Aer.get_backend("statevector_simulator")
    job = backend.run(circuit)
    result = job.result()

    output_state = result.get_statevector(circuit, decimals=3)
    state_distribution = output_state.probabilities().tolist()
    return state_distribution


def build_circuit(
    chromosome: List[Gate], qubit_num: int, input_gate: InputEncoding = None
) -> QuantumCircuit:
    circuit = QuantumCircuit(qubit_num)

    if input_gate is not None:
        circuit = input_gate.apply_to(circuit)

    for gate in chromosome:
        circuit = gate.apply_to(circuit)

    return circuit


def aggregate_state_distribution(
    state_distribution: List[float], qubit_num: int, ancillary_num: int
) -> List[float]:
    # This function assumes that ancillary qubits were added
    # after all "normal" qubits have been added.

    aggregated_distribution: List[float] = []
    for _ in range(2**qubit_num):
        aggregated_distribution.append(0.0)

    for i in range(2**qubit_num):
        for j in range(2**ancillary_num):
            aggregated_distribution[i] += state_distribution[i * 2**ancillary_num + j]

    return aggregated_distribution
