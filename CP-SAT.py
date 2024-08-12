from ortools.sat.python import cp_model


def generate_tsp_tour(tour_edges):
    # Initialize the tour starting with the first edge
    tour = []
    edge_dict = {start: end for start, end in tour_edges}
    # Start from the first node of the first edge
    current_node = tour_edges[0][0]

    while len(tour) < len(tour_edges) + 1:
        tour.append(current_node)
        current_node = edge_dict.get(current_node)

    return tour

def create_data_model(data_distance):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = data_distance
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def solve_tsp(data_distance):
    # Instantiate the data problem.
    data = create_data_model(data_distance)

    # Create the model.
    model = cp_model.CpModel()

    # Create the variables.
    n = len(data['distance_matrix'])
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = model.NewBoolVar(f'x[{i},{j}]')

    u = {}
    for i in range(n):
        u[i] = model.NewIntVar(0, n - 1, f'u[{i}]')

    # Create the objective function.
    objective = model.NewIntVar(0, 1000000, 'objective')
    model.Add(objective == sum(data['distance_matrix'][i][j] * x[i, j]
                               for i in range(n) for j in range(n) if i != j))
    model.Minimize(objective)

    # Create the constraints.
    # Each node must be visited exactly once
    for i in range(n):
        model.Add(sum(x[i, j] for j in range(n) if i != j) == 1)
        model.Add(sum(x[j, i] for j in range(n) if i != j) == 1)

    # Subtour elimination constraints
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.Add(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)

    # Initialize u variables
    for i in range(n):
        if i != 0:
            model.Add(u[i] > 0)
        model.Add(u[i] < n)
    model.Add(u[0] == 0)

    # Create the solver and solve the problem.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Print the solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Solution:')
        print(f'Objective value = {solver.Value(objective)}')
        tour = []
        for i in range(n):
            for j in range(n):
                if i != j and solver.Value(x[i, j]) > 0.5:
                    tour.append((i, j))
        tsp_route = generate_tsp_tour(tour)
        print('Tour:', " -> ".join(map(str, tsp_route)))
    else:
        print('The problem does not have an optimal solution.')


if __name__ == '__main__':
    # Read the distance matrix from the file
    file_name = 'att532'
    coords = []
    with open(file_name + '.txt') as f:
        for line in f.readlines():
            coords.append(list(map(float, [line.split()[-2], line.split()[-1]])))

    coord_dist = [[round(((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2) ** 0.5)
                   for j in range(len(coords))] for i in range(len(coords))]

    solve_tsp(coord_dist)
