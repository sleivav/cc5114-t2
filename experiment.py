import string
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from genetic_algorithm import StringProblemGeneticAlgorithm, Tournament
from maze import Maze


class Experiment:
    def __init__(self, population_sizes, mutation_rates, target, alphabet, genetic_algorithm_class, selection_function):
        self.population_sizes = population_sizes
        self.mutation_rates = mutation_rates
        self.target = target
        self.alphabet = alphabet
        self.averages = []
        self.maxes = []
        self.mins = []
        self.results = np.full((len(population_sizes), len(mutation_rates)), np.inf)
        self.genetic_algorithm_class = genetic_algorithm_class
        self.selection_function = selection_function

    def run_experiment(self, create_heatmap=True):
        for i, population_size in enumerate(self.population_sizes):
            for j, mutation_rate in enumerate(self.mutation_rates):
                genetic_algorithm_instance = self.genetic_algorithm_class(
                    self.target,
                    population_size,
                    mutation_rate,
                    self.selection_function,
                    self.alphabet
                )
                genetic_algorithm_instance.evaluate_fitness()
                self.run_iterations(genetic_algorithm_instance, i, j, 50, create_heatmap)

    def run_iterations(self, genetic_algorithm_instance, i, j, max_iterations=100, create_heatmap=True):
        iterations = 1
        while True:
            if not create_heatmap:
                fitness_list = genetic_algorithm_instance.fitness_list
                self.maxes += [max(fitness_list)]
                self.mins += [min(fitness_list)]
                self.averages += [sum(fitness_list) / len(fitness_list)]
            genetic_algorithm_instance.select()
            genetic_algorithm_instance.reproduce()
            genetic_algorithm_instance.evaluate_fitness()
            iterations += 1
            if iterations >= max_iterations:
                break
        if not create_heatmap:
            fitness_list = genetic_algorithm_instance.fitness_list
            self.maxes += [max(fitness_list)]
            self.mins += [min(fitness_list)]
            self.averages += [sum(fitness_list) / len(fitness_list)]
        if create_heatmap:
            self.results[i][j] = iterations

    def graph(self, heatmap=True):
        if heatmap:
            self.graph_heatmap()
        else:
            self.graph_evolution()

    def graph_heatmap(self):
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(self.results, norm=colors.LogNorm(vmin=self.results.min(), vmax=self.results.max()))
        column_labels = self.population_sizes
        row_labels = self.mutation_rates
        plt.locator_params(axis='y', nbins=len(column_labels))
        plt.locator_params(axis='x', nbins=len(row_labels))
        ax.set_xticklabels(row_labels, minor=False)
        ax.set_yticklabels(column_labels, minor=False)
        ax.set_xticks([float(n) + 0.5 for n in ax.get_xticks()])
        ax.set_yticks([float(n) + 0.5 for n in ax.get_yticks()])
        ax.set(xlim=(0, len(row_labels)), ylim=(0, len(column_labels)))
        plt.title('Número de iteraciones según cantidad de mutaciones\ny tamaño de la población')
        plt.xlabel('Ratio de mutaciones')
        plt.ylabel('Tamaño de la población')
        plt.colorbar(heatmap)
        plt.savefig('img/heatmap.png')
        plt.show()

    def graph_evolution(self):
        plt.plot(range(0, len(self.maxes)), self.maxes, label='Máximo por iteración')
        plt.plot(range(0, len(self.mins)), self.mins, label='Mínimo por iteración')
        plt.plot(range(0, len(self.averages)), self.averages, label='Promedio por iteración')
        plt.xlabel('Iteraciones')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crea un algoritmo genético y lo entrena para resolver un problema')
    parser.add_argument(
        'problem_type',
        type=str,
        default='binary',
        help='Tipo de problema a resolver, puede ser binary, alphanumeric o maze'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-heatmap', action='store_true')
    group.add_argument('-evolution_graph', action='store_true')
    args = parser.parse_args()
    population_sizes = []
    mutation_rates = []
    binary_word = ''
    alphabet_word = ''
    maze = None
    if args.heatmap:
        population_sizes = range(50, 1000, 50)
        mutation_rates = [float(x/10.0) for x in range(0, 10, 1)]
        binary_word = '10101010101010'
        alphabet_word = 'perritos'
        maze = Maze(5, 5)
    elif args.evolution_graph:
        population_sizes = [100]
        mutation_rates = [0.1]
        binary_word = '1010010101010100101010100101'
        alphabet_word = 'perritos y gatitos'
        maze = Maze(10, 10)
    if args.problem_type == 'binary':
        experiment = Experiment(population_sizes, mutation_rates, binary_word, '10', StringProblemGeneticAlgorithm, Tournament(5))
        experiment.run_experiment(args.heatmap)
        experiment.graph(args.heatmap)
    elif args.problem_type == 'alphabetic':
        experiment = Experiment(population_sizes, mutation_rates, alphabet_word, string.ascii_lowercase + ' ', StringProblemGeneticAlgorithm,
                                Tournament(5))
        experiment.run_experiment(args.heatmap)
    elif args.problem_type == 'maze':
        _, solution = maze.get_shortest_path()
        experiment = Experiment(population_sizes, mutation_rates, solution, 'UDLR', StringProblemGeneticAlgorithm,
                                Tournament(5))
        experiment.run_experiment(args.heatmap)
    print('done')
