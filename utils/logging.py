#!/usr/bin/env python3

from datetime import datetime
import os
import subprocess
from typing import List, Any

from gates import Gate
from ga import GA


def get_last_commit_id() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
    result = result.stdout.decode("utf-8")
    return result


def get_timestamp() -> str:
    return datetime.now().isoformat()


def log_experiment_details(
    ga: GA,
    experiment_id: str,
    target_path: str = "experiments.csv",
    description: str = "",
) -> None:
    header_row = "experiment_id; description; ga_params; gate_set; fitness; fitness_params; optimizer; optimizer_params; created_at; last_commit_id"

    if not os.path.exists(target_path):
        with open(target_path, "w") as target_file:
            target_file.write(header_row)

    created_at: str = get_timestamp()
    last_commit_id: str = get_last_commit_id()

    components = [
        experiment_id,
        description,
        str(ga.params),
        str(ga.gate_set),
        str(ga.fitness),
        str(ga.fitness.params),
        str(ga.optimizer),
        str(ga.optimizer.params),
        created_at,
        last_commit_id,
    ]

    with open(target_path, "a") as target_file:
        row = "; ".join(components)
        target_file.write("\n" + row)


def log_fitness(
    experiment_id: str,
    generation: int,
    best_fitness_value: float,
    mean_fitness_value: float,
    best_chromosome: List[Gate],
    target_path: str = "fitness_values.csv",
) -> None:
    header_row = "experiment_id; generation; best_fitness_value; mean_fitness_value; best_chromosome; created_at"

    if not os.path.exists(target_path):
        with open(target_path, "w") as target_file:
            target_file.write(header_row)

    created_at: str = get_timestamp()

    components = [
        experiment_id,
        str(generation),
        str(best_fitness_value),
        str(mean_fitness_value),
        str(best_chromosome),
        created_at,
    ]

    with open(target_path, "a") as target_file:
        row = "; ".join(components)
        target_file.write("\n" + row)


GATE_ADDED_EVENT: str = "GATE_ADDED_EVENT"
ALGORITHM_RESTART_EVENT: str = "ALGORITHM_RESTART_EVENT"


def log_event(
    experiment_id: str,
    event_type: str,
    payload: Any = None,
    target_path: str = "events.csv",
) -> None:
    header_row = "experiment_id; event_type; payload; created_at"

    if not os.path.exists(target_path):
        with open(target_path, "w") as target_file:
            target_file.write(header_row)

    created_at: str = get_timestamp()

    if payload is None:
        payload = ""
    elif type(payload) == str:
        payload = '"' + payload + '"'
    else:
        payload = str(payload)

    components = [
        experiment_id,
        event_type,
        payload,
        created_at,
    ]

    with open(target_path, "a") as target_file:
        row = "; ".join(components)
        target_file.write("\n" + row)
