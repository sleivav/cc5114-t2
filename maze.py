import random
from functools import reduce

import numpy as np


class Cell:
    def __init__(self, row: int, column: int, maze):
        self.row = row
        self.column = column
        self.visited = False
        self.left_wall = True
        self.right_wall = True
        self.top_wall = True
        self.bottom_wall = True
        self.maze = maze


class Maze:
    """
    Algoritmo de generación de laberintos, basado en https://en.wikipedia.org/wiki/Maze_generation_algorithm#Recursive_backtracker
    """
    def __init__(self, size_x, size_y):
        random.seed(123)
        self.size_x = size_x
        self.size_y = size_y
        self.cells = np.zeros((size_x, size_y), dtype=object)
        for row in range(size_x):
            for column in range(size_y):
                self.cells[row][column] = Cell(row, column, self)
        unvisited_cells = size_x * size_y
        current_cell = self.cells[0][0]
        stack = []
        while unvisited_cells > 0 and current_cell is not None:
            current_cell.visited = True
            current_neighbors = self.get_neighbors(current_cell.row, current_cell.column)
            unvisited_neighbors = list(filter(lambda cell: not cell.visited, current_neighbors))
            if len(unvisited_neighbors) > 0:
                next_cell = random.choice(unvisited_neighbors)
                if len(unvisited_neighbors) > 1:
                    stack.append(current_cell)
                self.remove_walls(current_cell, next_cell)
                current_cell = next_cell
                unvisited_cells -= 1
            elif stack:
                unvisited_neighbors_count = 0
                popped_cell = None
                while unvisited_neighbors_count == 0 and stack:
                    popped_cell = stack.pop()
                    current_neighbors = self.get_neighbors(popped_cell.row, popped_cell.column)
                    unvisited_neighbors_count = reduce(
                        lambda x, boolean: x + 1 if not boolean else x,
                        map(lambda cell: cell.visited, current_neighbors),
                        0
                    )
                    popped_cell.visited = True
                    unvisited_cells -= 1
                current_cell = popped_cell
        for row in range(size_x):
            for column in range(size_y):
                self.cells[row][column].visited = False

    """
    Dado una posición x e y en el laberinto, devuelve la lista de todos los vecinos de la celda correspondiente
    """
    def get_neighbors(self, current_row: int, current_column: int):
        neighbors = []
        if current_row != 0:
            neighbors.append(self.cells[current_row - 1][current_column])  # vecino arriba
        if current_column != 0:
            neighbors.append(self.cells[current_row][current_column - 1])  # vecino izquierda
        if current_row != self.size_x - 1:
            neighbors.append(self.cells[current_row + 1][current_column])  # vecino abajo
        if current_column != self.size_y - 1:
            neighbors.append(self.cells[current_row][current_column + 1])  # vecino derecha
        return neighbors

    @staticmethod
    def remove_walls(current_cell: Cell, next_cell: Cell):
        if current_cell.row == next_cell.row - 1:  # next_cell está abajo
            current_cell.bottom_wall = False
            next_cell.top_wall = False
        elif current_cell.row == next_cell.row + 1:  # next_cell está arriba
            current_cell.top_wall = False
            next_cell.bottom_wall = False
        elif current_cell.column == next_cell.column - 1:  # next_cell está a la derecha
            current_cell.right_wall = False
            next_cell.left_wall = False
        elif current_cell.column == next_cell.column + 1:  # next_cell está a la izquierda
            current_cell.left_wall = False
            next_cell.right_wall = False

    """
    Encuentra el camino más corto entre la celda que está arriba a la izquierda en un laberinto, y la celda que está
    abajo a la derecha
    """
    def get_shortest_path(self, current_row=0, current_column=0, dist=0, route=''):
        if current_row + 1 == self.size_x and current_column + 1 == self.size_y:
            return dist, route
        current_cell = self.cells[current_row][current_column]
        current_cell.visited = True

        if self.is_valid_movement(current_row + 1, current_column) and self.can_move_to(current_cell, current_row + 1, current_column):
            return self.get_shortest_path(current_row + 1, current_column, dist + 1, route + "D")
        if self.is_valid_movement(current_row - 1, current_column) and self.can_move_to(current_cell, current_row - 1, current_column):
            return self.get_shortest_path(current_row - 1, current_column, dist + 1, route + "U")
        if self.is_valid_movement(current_row, current_column + 1) and self.can_move_to(current_cell, current_row, current_column + 1):
            return self.get_shortest_path(current_row, current_column + 1, dist + 1, route + "R")
        if self.is_valid_movement(current_row, current_column - 1) and self.can_move_to(current_cell, current_row, current_column - 1):
            return self.get_shortest_path(current_row, current_column - 1, dist + 1, route + "L")
        current_cell.visited = False

    """
    Chequea si las nuevas pocisiones x e y son posibles de visitar, o se salen del laberinto
    """
    def is_valid_movement(self, x, y):
        return self.size_x > x >= 0 and self.size_y > y >= 0

    def can_move_to(self, cell, next_row, next_column):
        if self.cells[next_row][next_column].visited:
            return False
        elif cell.row == next_row - 1:  # intento de moverse abajo
            return not cell.bottom_wall
        elif cell.row == next_row + 1:  # intento de moverse arriba
            return not cell.top_wall
        elif cell.column == next_column - 1:  # intento de moverse hacia la derecha
            return not cell.right_wall
        elif cell.column == next_column + 1:  # intento de moverse hacie la izquierda
            return not cell.left_wall


if __name__ == '__main__':
    maze = Maze(10, 5)
    _, route = maze.get_shortest_path()
    print(route)
