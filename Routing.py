from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def main(coord_dist):
    def create_data_model(coord_dist):
        data = {}
        data["distance_matrix"] = coord_dist
        data["num_vehicles"] = 1
        data["depot"] = 0
        return data

    data = create_data_model(coord_dist)

    # 색인 관리자, 일종의 정보관리자
    manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])

    # Routing 객체 생성
    routing = pywrapcp.RoutingModel(manager)

    # 두 점 사이의 거리를 반환한다.
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    # 거리 계산 함수를 callback 함수로 등록하고 활용함.
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 시작해를 찾기 위한 휴리스틱 메소드를 등록
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)

    def print_solution(manager, routing, solution):
        print(f"Objective: {solution.ObjectiveValue()} miles")
        index = routing.Start(0)
        plan_output = "Route for vehicle 0:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += f" {manager.IndexToNode(index)} ->"
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += f" {manager.IndexToNode(index)}\n"
        print(plan_output)
        plan_output += f"Route distance: {route_distance}miles\n"

    # 문제 풀이
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        print_solution(manager, routing, solution)

    # 해 경로를 리스트에 저장하기 (Optional)
    def get_routes(solution, routing, manager):
        routes = []
        for route_nbr in range(routing.vehicles()):
            index = routing.Start(route_nbr)
            route = [manager.IndexToNode(index)]
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))
            routes.append(route)
        return routes

    routes = get_routes(solution, routing, manager)

if __name__ == '__main__':
    # Read the distance matrix from the file
    file_name = 'att532'
    coords = []
    with open(file_name + '.txt') as f:
        for line in f.readlines():
            coords.append(list(map(float, [line.split()[-2], line.split()[-1]])))

    coord_dist = [[round(((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2) ** 0.5)
                   for j in range(len(coords))] for i in range(len(coords))]

    main(coord_dist)