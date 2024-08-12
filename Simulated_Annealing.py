import numpy as np
import random
import math
from tqdm import tqdm
def total_distance(tour, distance_matrix):
    distance = 0
    num_cities =len(tour)

    for i in range(num_cities):
        distance += distance_matrix[tour[i]][tour[(i+1) % num_cities]]

    return distance

def swap_cities(tour):
    indices = range(1, len(tour))
    i,j = random.sample(indices,2)
    new_tour = tour[:]
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

    return new_tour


def simulated_annealing(distance_matrix, initial_temperature=1000000,cooling_rate=0.999,num_iterations=100000000,cut_off = 100):
    num_cities = len(distance_matrix)
    current_tour = list(range(1,num_cities))
    random.shuffle(current_tour)
    current_tour =  [0] + current_tour
    current_distance = total_distance(current_tour,distance_matrix)

    best_tour = current_tour
    best_distance = current_distance

    temperature = initial_temperature

    while best_distance > 100:
        new_tour = swap_cities(current_tour)
        new_disatnce = total_distance(current_tour,distance_matrix)

        if new_disatnce < current_distance or random.random() < math.exp((current_distance - new_disatnce) / temperature):
            currnet_tour = new_tour
            currnet_distance = new_disatnce

            if new_disatnce < best_distance:
                best_tour = new_tour
                best_distance = new_disatnce

        temperature *=cooling_rate

    return best_tour, best_distance


if __name__ == '__main__':
    # Read the distance matrix from the file
    file_name = 'ulysses16'
    coords = []
    with open(file_name + '.txt') as f:
        for line in f.readlines():
            coords.append(list(map(float, [line.split()[-2], line.split()[-1]])))

    coord_dist = [[(((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2))**0.5
                   for j in range(len(coords))] for i in range(len(coords))]

    best_tour, best_distance = simulated_annealing(coord_dist)

    print(f'best_distance={best_distance}')
    print(f'best_tour = {best_tour}')
