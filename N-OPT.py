import math

def length(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def N_OPT(input_data):
    # parse the input
    points = input_data
    nodeCount = len(points)

    distance = [[0] * nodeCount for _ in range(nodeCount)]

    for i in range(nodeCount):
        for j in range(i, nodeCount):
            distance[i][j] = length(points[i], points[j])
            distance[j][i] = distance[i][j]

    def cost_compute(cost_matrix, n1, n2, n3, n4):
        return cost_matrix[n1][n3] + cost_matrix[n2][n4] - cost_matrix[n1][n2] - cost_matrix[n3][n4]

    def compute_route_length(route, cost_matrix):
        total_length = 0
        for i in range(len(route)):
            total_length += cost_matrix[route[i - 1]][route[i]]
        return total_length

    def two_opt(cost_matrix, start_route):
        best_route = start_route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, nodeCount - 2):
                for j in range(i + 1, nodeCount):
                    if j - i == 1:
                        continue
                    if cost_compute(cost_matrix, best_route[i - 1], best_route[i], best_route[j - 1], best_route[j]) < 0:
                        best_route[i:j] = best_route[j - 1:i - 1:-1]
                        improved = True
        return best_route

    def three_opt(cost_matrix, start_route):
        best_route = start_route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, nodeCount - 3):
                for j in range(i + 2, nodeCount):
                    for k in range(j + 2, nodeCount):
                        if i == 0 and j == nodeCount - 1:
                            continue
                        new_route = (best_route[:i] +
                                     best_route[j:k+1][::-1] +
                                     best_route[i:j] +
                                     best_route[k+1:])
                        if compute_route_length(new_route, cost_matrix) < compute_route_length(best_route, cost_matrix):
                            best_route = new_route
                            improved = True
        return best_route

    route = list(range(nodeCount))
    #best_route = two_opt(distance, route)
    best_route = three_opt(distance, route)
    route_sum = compute_route_length(best_route, distance)

    # prepare the solution in the specified output format
    output_data = '%.2f %d\n' % (route_sum, 0)
    output_data += ' '.join(map(str, best_route))

    return output_data

if __name__ == '__main__':
    file_name = 'att532'
    coords = []

    with open(file_name + '.txt') as f:
        for line in f:
            coords.append(list(map(float, line.split()[-2:])))

    result = N_OPT(coords)
    print(result)
