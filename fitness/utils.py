#!/usr/bin/env python3

from qiskit import QuantumCircuit, Aer
from typing import List

from gates import Gate


def run_circuit(circuit: QuantumCircuit) -> List[float]:
    backend = Aer.get_backend("statevector_simulator")
    job = backend.run(circuit)
    result = job.result()

    output_state = result.get_statevector(circuit, decimals=3)
    state_distribution = output_state.probabilities().tolist()
    return state_distribution


def build_circuit(chromosome: List[Gate], qubit_num: int) -> QuantumCircuit:
    circuit = QuantumCircuit(qubit_num)

    for gate in chromosome:
        circuit = gate.apply_to(circuit)

    return circuit
