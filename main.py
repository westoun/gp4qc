#!/usr/bin/env python3

from experiments import (
    run_bell_state_2qubits,
    run_bell_state_3qubits,
    run_half_adder,
    run_deutsch,
    run_create_probability_distribution,
    run_grover,
    run_construct_relu,
    run_bernstein_vazirani,
)
from fitness import Fitness, BaselineFitness, IndirectQAFitness, DirectQAFitness

if __name__ == "__main__":
    # run_bell_state_2qubits()
    # run_bell_state_3qubits()
    # run_half_adder()
    # run_deutsch()
    # run_create_probability_distribution()
    run_grover(FitnessFunction=BaselineFitness)
    run_grover(FitnessFunction=IndirectQAFitness)
    run_grover(FitnessFunction=DirectQAFitness)
    run_grover(FitnessFunction=BaselineFitness, abstraction_learning=True)
    run_grover(FitnessFunction=IndirectQAFitness, abstraction_learning=True)
    run_grover(FitnessFunction=DirectQAFitness, abstraction_learning=True)
    # run_construct_relu()
    run_bernstein_vazirani(FitnessFunction=BaselineFitness)
    run_bernstein_vazirani(FitnessFunction=IndirectQAFitness)
    run_bernstein_vazirani(FitnessFunction=DirectQAFitness)
    run_bernstein_vazirani(FitnessFunction=BaselineFitness, abstraction_learning=True)
    run_bernstein_vazirani(FitnessFunction=IndirectQAFitness, abstraction_learning=True)
    run_bernstein_vazirani(FitnessFunction=DirectQAFitness, abstraction_learning=True)
