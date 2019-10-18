import math
import string
import random

import distance
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod, ABC

np.random.seed(5114)
random.seed(5144)


class SelectionFunction(ABC):
    @abstractmethod
    def select(self, fitness_list, population):
        pass


class Roulette(SelectionFunction):
    def select(self, fitness_list, population) -> list:
        probabilities = self.generate_probabilities(fitness_list)
        choices = np.random.choice(len(population), 2*len(population), p=probabilities)
        return [population[i] for i in choices]

    @staticmethod
    def generate_probabilities(fitness_list) -> list:
        total = sum(fitness_list)
        return [float(fitness_list[i]) / float(total) for i in range(len(fitness_list))]


class Tournament(SelectionFunction):
    def __init__(self, tournament_size=5):
        self.tournament_size = tournament_size

    def select(self, fitness_list, population) -> list:
        best_individual = population[fitness_list.index(max(fitness_list))]
        selection = [best_individual]
        for i in range(2*len(population) - 1):
            selection_indices = np.random.choice(len(population), self.tournament_size)
            fitnesses = [fitness_list[i] for i in selection_indices]
            max_fit_index = selection_indices[fitnesses.index(max(fitnesses))]
            selection += [population[max_fit_index]]
        return selection


class GeneticAlgorithm(ABC):
    def __init__(self, match_word: str, population_size: int, mutation_rate: float, selection_function: SelectionFunction, alphabet: str):
        self.match_word = match_word
        self.alphabet = alphabet
        self.population = [self.generate_individual() for m in range(population_size)]
        self.fitness_list = []
        self.selection_function = selection_function
        self.selection = []
        self.mutation_rate = mutation_rate

    def evaluate_fitness(self) -> None:
        fitness = []
        for individual in self.population:
            fitness += [self.get_ind_fitness(individual)]
        self.fitness_list = fitness

    @abstractmethod
    def solution_found(self) -> bool:
        pass

    '''
    Utiliza la selección de función para elegir los individuos que serán padres de la próxima generación
    '''
    def select(self):
        self.selection = self.selection_function.select(self.fitness_list, self.population)

    '''
    Utiliza los individuos seleccionados para generar la nueva generación
    '''
    def reproduce(self):
        for i in range(0, len(self.selection), 2):
            self.population[i // 2] = self.mutate(self.crossover(self.selection[i], self.selection[i+1]))

    '''
    Genera un nuevo individuo a partir de otros dos
    '''
    def crossover(self, ind1, ind2):
        index = np.random.randint(0, len(ind1))
        return ind1[0:index] + ind2[index:]

    '''
    Muta un individuo, cambio uno de sus genes por otro, al azar
    '''
    def mutate(self, ind):
        index = np.random.randint(0, len(ind), size=np.math.floor(len(ind) * self.mutation_rate))
        ind_list = list(ind)
        for i in index:
            ind_list[i] = random.choice(self.alphabet)
        return ''.join(ind_list)

    @abstractmethod
    def get_ind_fitness(self, individual):
        pass

    @abstractmethod
    def generate_individual(self):
        pass


class StringProblemGeneticAlgorithm(GeneticAlgorithm):
    """
    Para un algoritmo genético para strings, se calcula su fitness como la distancia de Levenshtein entre el gen y la
    palabra con la que se tiene que calzar
    """
    def get_ind_fitness(self, individual):
        return 0 - distance.levenshtein(individual, self.match_word)

    def solution_found(self) -> bool:
        return max(self.fitness_list) == 0

    def generate_individual(self) -> string:
        return ''.join([random.choice(self.alphabet) for _ in range(len(self.match_word))])


class LabyrinthGeneticAlgorithm(StringProblemGeneticAlgorithm):
    def generate_individual(self) -> string:
        return ''.join([random.choice(self.alphabet) for n in range(random.choice(range(100)))])

    def mutate(self, ind):
        functions = [self.remove_symbol, self.add_symbol, self.change_symbol]
        ind_list = list(ind)
        mutations_number = math.floor(len(ind) * self.mutation_rate)
        for i in range(mutations_number):
            index = np.random.randint(0, len(ind_list))
            ind_list = random.choice(functions)(index, ind_list)
        return ''.join(ind_list)

    @staticmethod
    def remove_symbol(index: int, ind_list: list):
        return ind_list[0:index] + ind_list[index + 1:]

    def add_symbol(self, index: int, ind_list: list):
        return ind_list[0:index] + [random.choice(self.alphabet)] + ind_list[index:]

    def change_symbol(self, index: int, ind_list: list):
        return ind_list[0:index-1] + [random.choice(self.alphabet)] + ind_list[index:]
