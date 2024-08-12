import math

def length(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def total_length(tour, points):
    """Calculate the total length of the tour including return to the starting point."""
    return sum(length(points[tour[i]], points[tour[i + 1]]) for i in range(len(tour) - 1)) + length(points[tour[-1]], points[tour[0]])

def nearest_neighbor(points):
    """Find a tour using the nearest neighbor heuristic."""
    n = len(points)
    unvisited = set(range(n))
    tour = [unvisited.pop()]

    while unvisited:
        last = tour[-1]
        next_point = min(unvisited, key=lambda x: length(points[last], points[x]))
        tour.append(next_point)
        unvisited.remove(next_point)

    return tour

def greedy(input_data):
    """Optimize tour using the nearest neighbor heuristic and return results in a specific format."""
    points = input_data
    optimized_tour = nearest_neighbor(points)
    total_dist = total_length(optimized_tour, points)
    output_data = '%.2f' % total_dist + ' ' + str(0) + '\n'  # Cost and 0 (unused parameter)
    output_data += ' '.join(map(str, optimized_tour)) + '\n'

    return output_data

if __name__ == '__main__':
    file_name = 'ulysses16'
    coords = []

    with open(file_name + '.txt') as f:
        for line in f.readlines():
            # Assuming the file's last two columns contain the coordinates
            coords.append(list(map(float, line.split()[-2:])))

    result = greedy(coords)
    print(result)