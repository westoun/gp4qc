#!/usr/bin/env python3

from typing import List, Union, Type

from gates import Gate, Oracle, CombinedGate, CombinedGateConstructor, OracleConstructor


def construct_gate_type_name(gate: Gate) -> str:
    """Construct the parameter-free name for a gate.
    Also handles gates with constructor classes.
    """
    if type(gate) in [CombinedGate, CombinedGateConstructor]:
        gate_names = []
        for GateType in gate.GateTypes:
            if GateType == Oracle or type(GateType) == OracleConstructor:
                gate_names.append("oracle")
            else:
                gate_names.append(GateType.name)

        return "_".join(gate_names)
    elif type(gate) in [Oracle, OracleConstructor]:
        # TA: How to handle multiple oracles?
        return "oracle"
    else:
        return gate.name


def construct_ngram_name(gates: List[Gate]) -> str:
    """Construct the ngram name for a list of gates
    based on their types.
    """

    gate_names = []
    for gate in gates:
        if type(gate) in [CombinedGate, CombinedGateConstructor]:
            for GateType in gate.GateTypes:
                if GateType == Oracle or type(GateType) == OracleConstructor:
                    gate_names.append("oracle")
                else:
                    gate_names.append(GateType.name)
        elif type(gate) in [Oracle, OracleConstructor]:
            # TA: How to handle multiple oracles?
            gate_names.append("oracle")
        else:
            gate_names.append(gate.name)
    return "_".join(gate_names)


def extract_ngram_types(
    gates: List[Union[Type, CombinedGateConstructor, OracleConstructor]]
) -> List[Type]:
    """Extract the (nested) gate types from a list of gates."""

    gate_types = []
    for gate in gates:
        if type(gate) == CombinedGateConstructor:
            gate_types.extend(gate.GateTypes)
        elif type(gate) == OracleConstructor:
            gate_types.append(gate)
        elif type(gate) in [Oracle, CombinedGate]:
            raise ValueError("Missing information! Should be constructor classes!")
        else:
            gate_types.append(gate)
    return gate_types


def get_unique_chomosomes(population: List[List[Gate]]) -> List[List[Gate]]:
    """Remove duplicate chromosomes based on gate types.
    Qubit ids or parameter values are not taken into account.
    If duplicates are encountered, the chromosome with the lower
    fitness value is chosen.
    """
    # for duplicates ,
    # keep the chromosomes with better (=lower) fitness.
    unique_chromosomes = {}
    for chromosome in population:
        chromosome_name = construct_ngram_name(chromosome)

        if chromosome_name in unique_chromosomes:
            if (
                unique_chromosomes[chromosome_name].fitness.values[0]
                > chromosome.fitness.values[0]
            ):
                unique_chromosomes[chromosome_name] = chromosome

        else:
            unique_chromosomes[chromosome_name] = chromosome

    return list(unique_chromosomes.values())
