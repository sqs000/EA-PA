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


# define the uniform crossover function
def crossover(population, p_c):
    # np.random.shuffle(population)
    for i in range(0,len(population) - (len(population)%2),2):
        if np.random.uniform(0, 1) < p_c:
            p1 = population[i]
            p2 = population[i+1]
            for i in range(len(p1)):
                if np.random.uniform(0, 1) < 0.5:
                    digit = p1[i]
                    p1[i] = p2[i]
                    p2[i] = digit
    return population

# define the mutation function, assume all the input is {0, 1}
def mutation(population, p_m):
    for p in population:
        for i in range(len(p)):
            if np.random.uniform(0, 1) < p_m:
                p[i] = 1 - p[i]
    return population

# define the selection function
def matingSelection(fitness, population):
    # scaling
    c_fmin = min(fitness)
    f_sum = sum(fitness) - (c_fmin - 0.001) * len(fitness)
    rw = [(fitness[0] - c_fmin + 0.001) / f_sum]
    for i in range(1, len(fitness)):
        rw.append(rw[i-1] + (fitness[i] - c_fmin + 0.001) / f_sum)

    # selected_fitness = []
    selected_population = []
    for i in range(len(population)) :
        r = np.random.uniform(0,1)
        index = 0
        # print(rw,r)
        while(r > rw[index]) :
            index = index + 1
        # selected_fitness.append(fitness[index])
        selected_population.append(population[index].copy())
    return selected_population


def s3674320_s3649024_GA(problem):
    # Parameters setting
    pop_size = 3
    p_c = 0.95 # 0.6, [0.75, 0.95]
    p_m = 0.02

    # Experiments
    # 15/0.5/0.02_3.5/46
    # 15/0.5/0.1_2.9/39
    # 15/0.6/0.02_3.62/46.5
    # 15/0.7/0.02_3.5/46
    # new_15/0.6/0.02_3.97/45

    # crossover experiment
    # 15/0.55/0.02_3.89/45.4
    # 15/0.60/0.02_3.97/45  **
    # 15/0.65/0.02_3.79/45.2
    # 15/0.70/0.02_3.72/45.1
    # 15/0.75/0.02_4.07/45.4
    # 15/0.77/0.02_3.76/45.5
    # 15/0.80/0.02_4.12/45.2 ***
    # 15/0.82/0.02_4.05/45.5
    # 15/0.85/0.02_3.84/45.8 
    # 15/0.90/0.02_3.96/45.5
    # 15/0.95/0.02_3.84/45.8 ***

    # mutation
    # 15/0.80/0.02_4.12/45.2
    # 15/0.80/0.03_3.82/45.8
    # 15/0.80/0.04_3.83/45.4
    # 15/0.80/0.05_3.93/45.5
    # 15/0.80/0.06_3.86/45.7
    # 15/0.80/0.08_3.74/45.7
    # 15/0.80/0.10_3.93/45.5


    # initial_pop = ... make sure you randomly create the first population
    population = np.random.choice([0, 1], size=(pop_size, dimension), replace=True)
    fitness = [problem(x) for x in population]
    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:

        # selection
        selected_pop = matingSelection(fitness, population)
        # crossover
        crossover_pop = crossover(selected_pop, p_c)
        # mutation 
        mutated_pop = mutation(crossover_pop, p_m)
        # reset
        # F18
        mutated_fitness = [problem(x) for x in mutated_pop]
        fit = np.concatenate((fitness, mutated_fitness), axis=0)
        pop = np.concatenate((population, mutated_pop), axis=0)
        sorted_indices = np.argsort(fit)[::-1]
        sorted_population = [pop[i] for i in sorted_indices]
        population = sorted_population[:pop_size]
        sorted_fitness = [fit[i] for i in sorted_indices]
        fitness = sorted_fitness[:pop_size]
        # F19
        # population = mutated_pop

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
        algorithm_name="GA_F19",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18)
    for run in range(20): 
        s3674320_s3649024_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    F19, _logger = create_problem(19)
    for run in range(20): 
        s3674320_s3649024_GA(F19)
        F19.reset()
    _logger.close()