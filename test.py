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

initial_pop = initialization(15)
print(encoding(initial_pop))