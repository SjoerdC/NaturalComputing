import tsplib95
import random
import matplotlib.pyplot as plt
import numpy
import time

from deap import base
from deap import creator
from deap import algorithms
from deap import tools

# Settings
instance = "burma14.tsp"
seed = 147644554
mutation_rate = 0.2

# Calculate the lenth of the tour
def path_length(path, data):

    length = 0;

    for n in range(1, len(path)-1):
        length += data.wfunc(path[n-1]+1, path[n]+1)

    return (length,)

def reverse_sequence_mutation(ind, memetic, data, icls):

    if random.random() <= mutation_rate:
        a = 0
        b = 1

        while a != b:
            a = random.randrange(0, len(ind), 1)
            b = random.randrange(0, len(ind), 1)

        val_a = ind[a]

        ind[a] = ind[b]
        ind[b] = val_a

    if memetic:
        return (icls(two_opt(ind, data)),)

    return (ind,)

# based on: https://github.com/rellermeyer/99tsp/blob/master/python/2opt/TSP2opt.py
def two_opt_swap(route, i, k):
    new_route = route[0:i]
    new_route.extend(reversed(route[i:k + 1]))
    new_route.extend(route[k+1:])

    return new_route

def two_opt(ind, data):

    best_route = ind

    while True:
        old_best  = path_length(best_route, data)
        best = path_length(best_route, data)
        for x in range(len(ind) - 1):
            for y in range(x+1, len(ind)):
                new = two_opt_swap(best_route, x, y)
                new_distance = path_length(new, data)

                if new_distance <= best:
                    best_route = new
                    best = new_distance

        if best <= old_best:
            return best_route

    return best_route

# show plot
def show_plot(logbook_normal, logbook_memetic):
    gen = logbook_normal.select("gen")
    fit_min_normal = logbook_normal.select("min")
    fit_avg_normal = logbook_normal.select("avg")
    fit_min_memetic = logbook_memetic.select("min")
    fit_avg_memetic = logbook_memetic.select("avg")

    plt.xlabel('Generations')
    plt.ylabel('Fitness')

    plt.plot(gen, fit_avg_normal, '--')
    plt.plot(gen, fit_min_normal, '--')
    plt.plot(gen, fit_avg_memetic)
    plt.plot(gen, fit_min_memetic)
    plt.legend(['avg fitness normal', 'min fitness normal', 'avg fitness memetic', 'min fitness memetic'], loc='upper left')

    plt.show()


def main():
    out_memetic = run(True)
    out_normal = run(False)

    show_plot(out_normal, out_memetic)

def run(memetic):

    # Set the seed
    random.seed(seed)

    # Receive the problem
    problem = tsplib95.load_problem("data/" + instance)
    nr_of_nodes = len(list(problem.get_nodes()))

    # Create the problem in DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("indices", random.sample, range(nr_of_nodes), nr_of_nodes)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", path_length, data=problem)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", reverse_sequence_mutation, memetic=memetic, data=problem, icls=creator.Individual)
    toolbox.register("select", tools.selTournament, tournsize=2)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)

    pop = toolbox.population(n=300)

    start_time = time.time()

    pop, out = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=1, ngen=100, stats=stats)

    print("Memetic: %r" % memetic)
    print("Execution time: %s seconds" % (time.time() - start_time))

    return out

if __name__ == "__main__":
    main()
