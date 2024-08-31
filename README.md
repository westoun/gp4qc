# Genetic Programming for Quantum Computing

**GP4QC** is a toolkit for generating quantum computing
circuits through the use of genetic algorithms.
It incorporates various fitness functions, optimizers,
as well as existing experiment implementations for a range
of problems.

## Installation

Before you get started, make sure you have
[Docker](https://www.docker.com/) installed.

To build the container, navigate into the project directory and execute

```bash
docker build --tag=gp4qc .
```

## Usage

Once the container has been built, you can run
it via

```bash
docker run -v /desired/local/path/to/results:/app/results -t gp4qc:latest &
```

The entrypoint of the container is the main.py-file
which specifies which experiments shall be run in
which configuration.
Experiment-specific parameters, such as the population size
of the genetic algorithm or crossover probability,
have to be specified in the corresponding file within the
experiments directory.

In the case of the Grover and Bernstein-Vazirani experiments,
callbacks have been implemented that log the mean fitness score
of a population after each generation. This data is then written
to a csv file in the volumne attached to the container.
If you wish to run the code on your system, make sure to adjust the
local path of the volumne definition.

## Experiment Structure

The following lines illustrate the code structure for
the creation of a bell state with 2 qubits.
If you wish to add a new experiment, make sure to
include all of the required components.

```python

# Specify which gates the genetic algorithm is allowed to use.
gate_set: GateSet = GateSet(
    gates=[H, X, Y, Z, CX, CY, CZ, Identity], qubit_num=2
)

# Specify what state distributions shall be approximated.
target_distributions: List[List[float]] = [[0.5, 0, 0, 0.5]]

# Specify the parameters of the genetic algorithm.
ga_params = GAParams(
    population_size=100,
    generations=20,
    crossover_prob=0.5,
    swap_gate_mutation_prob=0.1,
    chromosome_length=4
)

# Choose and instantiate one of the available fitness functions.
fitness: Fitness = Jensensshannon()

# Provide context to the optimizer through parameters.
optimizer_params = OptimizerParams(qubit_num=2, measurement_qubit_num=2)

# Choose and instantiate one of the available optimizer classes. If parameterized gates are used, a corresponding optimizer might f.e. optimize parameter values before determining the final fitness score of a chromosome.
optimizer: Optimizer = DoNothingOptimizer(target_distributions, params = optimizer_params)

# Instantiate and run the genetic algorithm.
genetic_algorithm = GA(gate_set, fitness, optimizer, params=ga_params)
genetic_algorithm.run()

# Register a callback function that gets triggered after each generation.
genetic_algorithm.on_after_generation(log_fitness_callback)
```

## Contributing

Pull requests are welcome. For major changes, please
open an issue first to discuss what you would like to
change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
