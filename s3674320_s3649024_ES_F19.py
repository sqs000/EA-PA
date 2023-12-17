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
    initial_sigma = []
    for i in range(mu):
        initial_sigma.append(0.03 * (31 - 0))
    return parent, np.array(initial_sigma)


# Encoding 
def encoding(population):
    new_population = []
    for individual in population:
        individual_segmented = [individual[i:i + 5] for i in range(0, len(individual), 5)]
        new_individual = [list(map(str, int_list)) for int_list in individual_segmented]
        for i, encoded_list in enumerate(new_individual):
            new_individual[i] = int("".join(encoded_list), 2)
        new_population.append(new_individual)
    return np.array(new_population)


# Decoding
def decoding(population):
    decoded_population = []
    for individual in population:
        decoded_individual = []
        for encoded_value in individual:
            binary_string = bin(encoded_value)[2:].zfill(5)  # Convert to binary and fill with leading zeros
            decoded_individual.extend(list(map(int, binary_string)))
        decoded_population.append(decoded_individual)
    return np.array(decoded_population)


# Discrete recombination, use intermediate sigma
def recombination(population, parent_sigma, num_offsprings):
    offsprings = []
    offsprings_sigma = []
    for _ in range(num_offsprings):
        [p1,p2] = np.random.choice(len(population), 2, replace = False)
        offspring = []
        for i in range(len(population[p1])):
            offspring.append(np.random.choice([population[p1][i], population[p2][i]]))
        offsprings.append(offspring)
        offsprings_sigma.append((parent_sigma[p1] + parent_sigma[p2])/2)
    return np.array(offsprings), np.array(offsprings_sigma)


# One-sigma mutation
def mutation(population, sigma, tau):
    mutated_population = np.copy(population)
    mutated_sigma = np.copy(sigma)
    for individual, individual_sigma in zip(mutated_population, mutated_sigma):
        individual_sigma = individual_sigma * np.exp(np.random.normal(0, tau))
        for i in range(len(individual)):
            individual[i] = int(individual[i] + np.random.normal(0, individual_sigma))
            individual[i] = individual[i] if individual[i] <= 31 else 31
            individual[i] = individual[i] if individual[i] >= 0 else 0
    return mutated_population, mutated_sigma


# (mu + lambda) selection
def selection(parent_pop, offspring_pop, parent_sigma, offspring_sigma, problem, mu):
    # mu + lambda
    population = np.concatenate((parent_pop, offspring_pop), axis=0)
    sigma = np.concatenate((parent_sigma, offspring_sigma))
    # evaluation and sort
    fitness_values = [problem(x) for x in population]
    sorted_indices = np.argsort(fitness_values)[::-1]
    sorted_fitness = [fitness_values[i] for i in sorted_indices]
    sorted_population = [population[i] for i in sorted_indices]
    sorted_sigma = [sigma[i] for i in sorted_indices]
    # select
    selected_population = sorted_population[:mu]
    selected_sigma = sorted_sigma[:mu]
    selected_fitness = sorted_fitness[:mu]
    return np.array(selected_population), np.array(selected_sigma), np.array(selected_fitness)


def s3674320_s3649024_ES(problem):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    # initial_pop = ... make sure you randomly create the first population
    mu_ = 50
    lambda_ = 500
    initial_pop, initial_sigma = initialization(mu_)
    tau_0 =  1.0 / np.sqrt(len(initial_pop[0])/5)
    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        # encoding
        encoded_pop = encoding(initial_pop)
        # recombination
        recombined_pop, recombined_sigma = recombination(encoded_pop, initial_sigma, lambda_)
        # mutation
        mutated_pop, mutated_sigma = mutation(recombined_pop, recombined_sigma, tau_0)
        # decoding
        decoded_pop = decoding(mutated_pop)
        # selection
        selected_pop, selected_sigma, fitness_values = selection(initial_pop, decoded_pop, initial_sigma, mutated_sigma, problem, mu_)
        # reset
        initial_pop = selected_pop
        initial_sigma = selected_sigma
    # show best individual in the end
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
        algorithm_name="ES_F19",  # name of your algorithm
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


