import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass


budget = 5000
dimension = 50
np.random.seed(42)
# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`


# Initialization
def initialization(mu):
    parent = np.random.choice([0, 1], size=(mu, dimension), replace=True)
    return parent       


# Discrete recombination
def recombination(parent, num_offsprings):
    offsprings = []
    for _ in range(num_offsprings):
        [p1,p2] = np.random.choice(len(parent), 2, replace = False)
        offspring = []
        for i in range(dimension):
            offspring.append(np.random.choice([parent[p1][i], parent[p2][i]]))
        offsprings.append(offspring)
    return np.array(offsprings)


# Bit flip mutation
def mutation(population, mutation_probability):
    mutated_population = np.copy(population)
    for individual in mutated_population:
        for i in range(dimension):
            if np.random.rand() < mutation_probability:
                individual[i] = 1 - individual[i]
    return mutated_population


# (mu + lambda) selection
def selection(parent_pop, offspring_pop, problem, mu):
    population = np.concatenate((parent_pop, offspring_pop), axis=0)
    fitness_values = [problem(x) for x in population]
    sorted_indices = np.argsort(fitness_values)[::-1]
    sorted_population = [population[i] for i in sorted_indices]
    selected_population = sorted_population[:mu]
    sorted_fitness = [fitness_values[i] for i in sorted_indices]
    selected_fitness = sorted_fitness[:mu]
    return selected_population, selected_fitness


def s3674320_s3649024_ES(problem):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    # initial_pop = ... make sure you randomly create the first population
    mu_ = 15
    lambda_ = 100
    mutation_rate = 0.1
    initial_pop = initialization(mu_)
    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        # recombination
        recombined_pop = recombination(initial_pop, lambda_)
        # mutation
        mutated_pop = mutation(recombined_pop, mutation_rate)
        # selection
        selected_pop, fitness_values = selection(initial_pop, mutated_pop, problem, mu_)
        # reset
        initial_pop = selected_pop
    print(selected_pop[0], fitness_values[0])
    # no return value needed 


def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="bit-flip_(mu+lambda)_ES",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18)
    for run in range(20): 
        s3674320_s3649024_ES(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    F19, _logger = create_problem(19)
    for run in range(20): 
        s3674320_s3649024_ES(F19)
        F19.reset()
    _logger.close()


