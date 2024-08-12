import numpy as np
import matplotlib.pyplot as plt
import random


def calculate_distance(city1, city2):
    return np.linalg.norm(city1 - city2)


def calculate_total_distance(tour, coords):
    total_distance = 0
    for i in range(len(tour)):
        total_distance += calculate_distance(coords[tour[i]], coords[tour[(i + 1) % len(tour)]])
    return total_distance


def create_individual(num_cities):
    individual = list(range(num_cities))
    random.shuffle(individual)
    return individual


def create_population(pop_size, num_cities):
    return [create_individual(num_cities) for _ in range(pop_size)]


def select(population, fitnesses):
    fitnesses_sum = fitnesses.sum()
    if fitnesses_sum == 0:
        fitnesses = np.ones_like(fitnesses)
    else:
        fitnesses = fitnesses / fitnesses_sum
    return np.array([population[i] for i in np.random.choice(len(population), size=len(population), p=fitnesses)])


def crossover(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]
    pos = end
    for gene in parent2:
        if gene not in child:
            if pos >= size:
                pos = 0
            child[pos] = gene
            pos += 1
    return child


def mutate(individual):
    size = len(individual)
    i, j = random.sample(range(size), 2)
    individual[i], individual[j] = individual[j], individual[i]


def alternative_mutate(individual):
    size = len(individual)
    i, j = random.sample(range(size), 2)
    if i > j:
        i, j = j, i
    individual[i:j + 1] = reversed(individual[i:j + 1])


def adaptive_mutation_rate(generation, max_generations):
    initial_mutation_rate = 0.2
    decay = 0.05
    return max(0.01, initial_mutation_rate * (1 - (generation / max_generations) * decay))


def enhanced_elitism(population, elite_size, coords):
    sorted_population = sorted(population, key=lambda ind: calculate_total_distance(ind, coords))
    return sorted_population[:elite_size]


def genetic_algorithm(coords, pop_size=100, generations=10000, elite_size=0.2):
    num_cities = len(coords)
    population = create_population(pop_size, num_cities)
    elite_count = int(elite_size * pop_size)
    best_individual = min(population, key=lambda ind: calculate_total_distance(ind, coords))

    for generation in range(generations):
        fitnesses = np.array([1 / calculate_total_distance(ind, coords) for ind in population])
        selected = select(population, fitnesses)

        new_population = enhanced_elitism(population, elite_count, coords)  # Use enhanced elitism

        mutation_rate = adaptive_mutation_rate(generation, generations)

        for _ in range(pop_size - elite_count):
            parent1, parent2 = random.sample(list(selected), 2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                alternative_mutate(child)  # Use alternative mutation
            new_population.append(child)

        population = new_population
        current_best = min(population, key=lambda ind: calculate_total_distance(ind, coords))
        if calculate_total_distance(current_best, coords) < calculate_total_distance(best_individual, coords):
            best_individual = current_best

        if generation % 50 == 0:
            print(f"Generation {generation}: Distance = {calculate_total_distance(best_individual, coords)}")

    return best_individual, calculate_total_distance(best_individual, coords)


def random_restart_genetic_algorithm(coords, restarts=10, **kwargs):
    best_solution = None
    best_distance = float('inf')

    for _ in range(restarts):
        print("Restarting...")
        solution, distance = genetic_algorithm(coords, **kwargs)
        if distance < best_distance:
            best_solution = solution
            best_distance = distance

    return best_solution, best_distance


def plot_solution(coords, tour):
    plt.figure(figsize=(10, 7))
    tour_coords = np.array([coords[city] for city in tour] + [coords[tour[0]]])
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'o-')
    plt.title("TSP Solution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Read the coordinates from the file
    file_name = 'att48'
    coords = []
    with open(file_name + '.txt') as f:
        for line in f.readlines():
            coords.append(list(map(float, [line.split()[-2], line.split()[-1]])))

    coords = np.array(coords)
    best_tour, best_distance = random_restart_genetic_algorithm(coords, restarts=1, pop_size=50, generations=200000,
                                                                elite_size=0.2)

    print(f"Best Tour: {list(best_tour)}")
    print(f"Best Distance: {best_distance}")

    plot_solution(coords, best_tour)
