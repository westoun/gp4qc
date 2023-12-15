#!/usr/bin/env python3

from qiskit import QuantumCircuit, Aer, transpile
from qiskit_aer import AerError
from typing import List

from gates import Gate, MultiCaseGate, InputEncoding, Oracle


def run_circuit(circuit: QuantumCircuit, decompose_reps: int = 5) -> List[float]:
    backend = Aer.get_backend("statevector_simulator")

    transpiled_circuit = circuit.decompose(reps=decompose_reps)

    job = backend.run(transpiled_circuit)

    try:
        result = job.result()
    except AerError as e:
        if decompose_reps > 10:  # avoid infinite retries
            raise AerError(str(e))

        return run_circuit(circuit, decompose_reps=decompose_reps + 1)

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
