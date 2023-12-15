#!/usr/bin/env python3

from qiskit import QuantumCircuit, Aer, transpile
from typing import List

from gates import Gate, MultiCaseGate, InputEncoding, Oracle


def run_circuit(circuit: QuantumCircuit) -> List[float]:
    backend = Aer.get_backend("statevector_simulator")

    transpiled_circuit = circuit.decompose(reps=2)

    job = backend.run(transpiled_circuit)
    result = job.result()

    output_state = result.get_statevector(transpiled_circuit, decimals=3)
    state_distribution = output_state.probabilities().tolist()
    return state_distribution


def build_circuit(
    chromosome: List[Gate],
    qubit_num: int,
    case_index=0,
) -> QuantumCircuit:
    circuit = QuantumCircuit(qubit_num)

    for gate in chromosome:
        if issubclass(gate.__class__, MultiCaseGate):
            gate.set_case_index(case_index)

        circuit = gate.apply_to(circuit)

    return circuit


def aggregate_state_distribution(
    state_distribution: List[float], measurement_qubit_num: int, ancillary_num: int
) -> List[float]:
    # This function assumes that ancillary qubits were added
    # after all "normal" qubits have been added.

    aggregated_distribution: List[float] = []
    for _ in range(2**measurement_qubit_num):
        aggregated_distribution.append(0.0)

    for i in range(2**measurement_qubit_num):
        for j in range(2**ancillary_num):
            aggregated_distribution[i] += state_distribution[i * 2**ancillary_num + j]

    return aggregated_distribution
