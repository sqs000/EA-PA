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
    for i in range(0,len(population) - (len(population)%2),2):
        p1 = population[i]
        p2 = population[i+1]
        if np.random.uniform(0, 1) < p_c:
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
    f_sum = sum(fitness) - c_fmin * len(fitness)
    rw = [(fitness[0] - c_fmin) / f_sum]
    for i in range(1, len(fitness)):
        rw.append(rw[i-1] + (fitness[i] - c_fmin) / f_sum)

    selected_population = []
    for i in range(len(population)) :
        r = np.random.uniform(0,1)
        index = 0
        # print(rw,r)
        while(r > rw[index]) :
            index = index + 1
        selected_population.append(population[index].copy())
    return selected_population


def s3674320_s3649024_GA(problem):
    # Parameters setting
    pop_size = 100
    p_c = 0.6 # 0.6, [0.75, 0.95]
    p_m = 0.02

    # Experiments
    # 15/0.5/0.02_3.5/46
    # 15/0.5/0.1_2.9/39
    # 15/0.6/0.02_3.6/46.5
    # 15/0.7/0.02_3.5/46

    # initial_pop = ... make sure you randomly create the first population
    population = np.random.choice([0, 1], size=(pop_size, dimension), replace=True)

    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        # selection
        fitness = [problem(x) for x in population]
        selected_pop = matingSelection(fitness, population)
        # crossover
        crossover_pop = crossover(selected_pop, p_c)
        # mutation 
        mutated_pop = mutation(crossover_pop, p_m)
        # reset
        population = mutated_pop

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
        algorithm_name="genetic_algorithm",  # name of your algorithm
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