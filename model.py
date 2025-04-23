import numpy as np
from numba import njit, prange
height = 200
width = 200


grid_model = np.zeros((height, width), dtype=int)
next_grid_model = np.zeros((height, width), dtype=int)
neighbors = np.zeros((height, width), dtype=int)


@njit(parallel=True, cache=True, nogil=True)
def randomize(grid, width, height):
    grid[:] = np.random.randint(0, 2, size=(height, width))


@njit(parallel=True, cache=True, nogil=True)
def next_gen(grid_model, next_grid_model, neighbors, height, width):

    # Обработаем соседей для каждой клетки
    neighbors.fill(0)

    for i in prange(height):
        for j in prange(width):
            neighbors[i, j] = count_neighbors(grid_model, i, j, height, width)

    # Применим правила игры
    next_grid_model.fill(0)  # Обнулим grid_model

    for i in prange(height):
        for j in prange(width):
            if grid_model[i, j] == 0 and neighbors[i, j] == 3:
                next_grid_model[i, j] = 1
            elif grid_model[i, j] == 1 and (neighbors[i, j] == 2 or neighbors[i, j] == 3):
                next_grid_model[i, j] = 1

    grid_model[:, :] = next_grid_model


def next_gen_short():
    next_gen(grid_model, next_grid_model, neighbors, height, width)


@njit(parallel=True, cache=True, nogil=True)
def count_neighbors(grid, row, col, height, width):
    count = 0
    # Определим соседей
    for r in prange(row-1, row+2):
        for c in prange(col-1, col+2):
            if (r == row and c == col) or r < 0 or r >= height or c < 0 or c >= width:
                continue
            count += grid[r, c]
    return count


glider_pattern = [[0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]]


glider_gun_pattern = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                       0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                       0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                       0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
                       0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


def load_pattern(pattern, x_offset=0, y_offset=0):
    global grid_model

    for i in range(0, height):
        for j in range(0, width):
            grid_model[i][j] = 0

    j = y_offset

    for row in pattern:
        i = x_offset
        for value in row:
            grid_model[i][j] = value
            i = i + 1
        j = j + 1
