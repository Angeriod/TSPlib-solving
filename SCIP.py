from ortools.linear_solver import pywraplp

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

def solve_tsp(DIST):
    nCity = len(DIST)
    solver = pywraplp.Solver.CreateSolver("SCIP")
    X = {}
    for i in range(nCity):
        for j in range(nCity):
            if i != j:
                X[i, j] = solver.IntVar(0, 1, "X"+str(i)+str(j))

    U = {}
    for i in range(1, nCity):
        U[i] = solver.IntVar(1, nCity-1, "U[%i]" %i)



    for j in range(nCity):
        solver.Add(sum([X[i,j] for i in range(nCity) if i!=j]) == 1)


    for i in range(nCity):
        solver.Add(sum([X[i,j] for j in range(nCity) if i!=j]) == 1)



    for i in range(1, nCity):
        for j in range(1, nCity):
            if i!=j:
                solver.Add(U[i]-U[j] + nCity*X[i,j] <= nCity-1)


    # Objective
    objective_terms = []
    for i in range(nCity):
        for j in range(nCity):
            if i != j:
                objective_terms.append(DIST[i][j] * X[i, j])

    solver.Minimize(solver.Sum(objective_terms))


    if 1:
        with open('or9-1.lp', "w") as out_f:
            lp_text = solver.ExportModelAsLpFormat(False)
            out_f.write(lp_text)
    # Solve
    status = solver.Solve()

    # Print solution.
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Total cost = {solver.Objective().Value():.1f}\n", )
        tour = []
        for i in range(nCity):
            for j in range(nCity):
                if i != j:
                    if X[i, j].solution_value() > 0.5:
                        tour.append((i, j))
        tsp_route = generate_tsp_tour(tour)
        print('Tour:', " -> ".join(map(str, tsp_route)))
    else:
        print("No solution found.")

if __name__ == '__main__':
    file_name = 'ulysses16'
    DIST = []
    with open(file_name + '.txt') as f:
        for line in f.readlines():
            DIST.append(list(map(float, [line.split()[-2], line.split()[-1]])))

    DIST = [[round(((DIST[i][0] - DIST[j][0]) ** 2 + (DIST[i][1] - DIST[j][1]) ** 2) ** 0.5)
                       for j in range(len(DIST))] for i in range(len(DIST))]

    solve_tsp(DIST)