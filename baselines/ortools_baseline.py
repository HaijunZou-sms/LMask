import os
import sys
import time
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, os.pardir))
sys.path.append(project_root)
from tensordict.tensordict import TensorDict
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from baselines.utils import scale

ORTOOLS_SCALING_FACTOR = 1000


def ortools_solve(td: TensorDict, max_runtime: float = 100, log_search: bool = False):
    data = make_ortools_data(td)
    distance_matrix = data["distance_matrix"]
    num_nodes = len(distance_matrix)

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(num_nodes, data["num_vehicles"], data["depot"])
    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # register a distance transit callback
    distance_transit_idx = routing.RegisterTransitMatrix(data["distance_matrix"])
    routing.SetArcCostEvaluatorOfAllVehicles(distance_transit_idx)

    if "draft_limit" in td.keys():
        # TODO: There are some issues with the draft limit constraint
        demand = data["demand"]
        draft_limit = data["draft_limit"]
        # Create and register a demand callback
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demand[from_node]
        demand_callback_idx = routing.RegisterUnaryTransitCallback(demand_callback) 
        # Add load dimension
        routing.AddDimension(
            demand_callback_idx,
            0, # no slack
            sum(demand),
            True,
            "Load",
        )
        load_dimension = routing.GetDimensionOrDie("Load")
        
        # Add draft limit constraint for each location excluding the depot
        for node in range(0, num_nodes):
            index = manager.NodeToIndex(node)
            # Current load when arriving at this node must <= its draft limit
            load_dimension.CumulVar(index).SetRange(0, draft_limit[node])
    
    if "time_windows" in data.keys(): 
        # create a time transit callback
        depot_tw_late = data["time_windows"][0][1]  # depot time window is [0, depot_tw_late]

        def time_transit_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node] + data["service_time"][from_node]
        time_transit_callback_idx = routing.RegisterTransitCallback(time_transit_callback)
        routing.AddDimension(
            time_transit_callback_idx,
            depot_tw_late,  # waiting time upper bound
            depot_tw_late,  # maximum duration per vehicle
            False,  # Don't force start cumul to zero at depot
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        # Add time window constraints for each location including the depot
        for node in range(1, num_nodes):
            index = manager.NodeToIndex(node)
            tw_early, tw_late = data["time_windows"][node]
            time_dim.CumulVar(index).SetRange(tw_early, tw_late)

        # minimize the global makespan time
        for i in range(data["num_vehicles"]):
            routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(i)))

    # Set First Solution Heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.log_search = log_search
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = max_runtime

    # Solve the problem
    start_time = time.perf_counter()
    solution = routing.SolveWithParameters(search_parameters)
    duration = time.perf_counter() - start_time
    if solution:
        #print_solution(manager, data, routing, solution)
        action = solution2action(data, manager, routing, solution)
        cost = solution.ObjectiveValue() / ORTOOLS_SCALING_FACTOR
        return action, cost, duration
    else:
        print(f"No solution found! Satus: {routing.status()} ")
        return np.arange(num_nodes), np.inf, duration


def make_ortools_data(td: TensorDict, scaling_factor: int = ORTOOLS_SCALING_FACTOR):
    num_locs = td["locs"].size(-2) - 1  # Exclude depot
   
    data_dict = {}
    data_dict["distance_matrix"] = scale(td["distance_matrix"], scaling_factor).tolist()

    if "draft_limit" in td.keys():
        td["demand"] = td["demand"] * num_locs
        td["draft_limit"] = td["draft_limit"] * num_locs
        data_dict["demand"] = scale(td["demand"], scaling_factor).tolist()
        data_dict["draft_limit"] = scale(td["draft_limit"], scaling_factor).tolist()

    if "time_windows" in td.keys():
        data_dict["time_windows"] = scale(td["time_windows"], scaling_factor).tolist()
        data_dict["service_time"] = scale(td["service_time"], scaling_factor).tolist()
    data_dict["num_vehicles"] = 1
    data_dict["depot"] = 0
   
    return data_dict


def print_solution(manager, data, routing, solution, load_dimension=None):
    """Prints solution on console."""
    if not solution:
        print("No solution found!")
        return

    print(f"Objective: {solution.ObjectiveValue()}")

    index = routing.Start(0)
    plan_output = "Route:\n"
    route_distance = 0
    load_values = []

    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        plan_output += f" {node} ->"
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

        if load_dimension:
            load_values.append(solution.Value(load_dimension.CumulVar(previous_index)))

    # Add the depot at the end
    node = manager.IndexToNode(index)
    plan_output += f" {node}\n"
    plan_output += f"Total distance: {route_distance}\n"

    print(plan_output)

    if load_dimension:
        print("\nDraft Limit Compliance Check:")
        index = routing.Start(0)
        while True:
            node = manager.IndexToNode(index)
            actual_load = solution.Value(load_dimension.CumulVar(index))
            draft_limit = data['draft_limit'][node] if 'draft_limit' in data else float('inf')
            
            status = "OK" if actual_load <= draft_limit else "VIOLATION"
            print(f"Node {node}: Load={actual_load} (Limit={draft_limit}) -> {status}")
            
            if routing.IsEnd(index):
                break
            index = solution.Value(routing.NextVar(index))


def solution2action(data, manager, routing, solution) -> list[list[int]]:
    """
    Converts an OR-Tools solution to routes.
    """
    routes = []
    distance = 0  # for debugging

    for vehicle_idx in range(data["num_vehicles"]):
        index = routing.Start(vehicle_idx)
        route = []
        route_cost = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)

            prev_index = index
            index = solution.Value(routing.NextVar(index))
            route_cost += routing.GetArcCostForVehicle(prev_index, index, vehicle_idx)

        if clients := route[1:]:  # ignore depot
            routes.append(clients)
            distance += route_cost

    return [visit for route in routes for visit in route + [0]]


if __name__ == "__main__":
    # Example usage
    import torch
    from rl4co.data.utils import load_npz_to_tensordict

    file_path = "../data/random/tsptw/test/tsptw50_test_easy_seed2025.npz"
    td = load_npz_to_tensordict(file_path)
    td = td[1]
    distance_matrix = torch.cdist(td["locs"], td["locs"], p=2)
    distance_matrix.diagonal(dim1=-2, dim2=-1).fill_(0)
    td.update({"distance_matrix": distance_matrix})
    action, cost, duration = ortools_solve(td, max_runtime=20, log_search=True)
    print(f"Action: {action}, Cost: {cost}, Duration: {duration}")
