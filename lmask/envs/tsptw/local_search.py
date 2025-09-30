import pyvrp
from pyvrp import (
    Client,
    CostEvaluator,
    Depot,
    ProblemData,
    RandomNumberGenerator,
    Solution,
    VehicleType,
)
from pyvrp.search import (
    Exchange10,
    LocalSearch,
    NeighbourhoodParams,
    compute_neighbours,
    NODE_OPERATORS,
    ROUTE_OPERATORS,
)
from typing import Tuple, Union
from multiprocessing import Pool
import torch
import numpy as np
from tensordict.tensordict import TensorDict
from torch import Tensor

PYVRP_SCALING_FACTOR = 1_000
MAX_VALUE = 1 << 42


def local_search_from_data(
    instances: TensorDict,
    solutions,
    max_trials=1,
    num_workers=0,
    tw_penalty=0.2,
    return_only_solutions=True,
):
    """
    Parameters:
    instance  (batch_size, ~)
    solutions (batch_size, aug, ~)
    return_only_solutions: If True, only return improved solutions. If False, return solutions, costs, and feasibility.
    """
    instances["distance_matrix"].diagonal(dim1=-2, dim2=-1).fill_(0)
    if isinstance(solutions, torch.Tensor):
        solutions = solutions.cpu().numpy()
    improved_solutions = []
    costs = []
    feasibility = []

    if num_workers > 0:
        with Pool(num_workers) as pool:
            results = pool.starmap(
                local_search_per_instance,
                [(instance, solutions[idx], max_trials, tw_penalty) for idx, instance in enumerate(instances)],
            )
            for instance_improved_solutions, instance_costs, instance_feasibility in results:
                improved_solutions.extend(instance_improved_solutions)
                costs.extend(instance_costs)
                feasibility.extend(instance_feasibility)
    else:
        for idx, instance in enumerate(instances):
            instance_improved_solutions, instance_costs, instance_feasibility = local_search_per_instance(instance, solutions[idx], max_trials)
            improved_solutions.extend(instance_improved_solutions)
            costs.extend(instance_costs)
            feasibility.append(instance_feasibility)

    if return_only_solutions:
        return torch.LongTensor(improved_solutions)
    return (
        torch.LongTensor(improved_solutions),
        torch.tensor(costs),
        torch.tensor(feasibility, dtype=torch.bool),
    )


def local_search_per_instance(instance, instance_solutions, max_trials, tw_penalty=0.01):

    # make data and search operator in advance
    problem_data = make_pyvrp_data(instance)
    ls = make_search_operator(problem_data)

    instance_improved_solutions = []
    instance_costs = []
    instance_feasibility = []
    if len(instance_solutions.shape) == 1:
        instance_solutions = np.expand_dims(instance_solutions, axis=0)
    for path_solution in instance_solutions:
        solution = Solution(problem_data, [path_solution])
        
        improved_solution = perform_local_search(ls, solution, max_trials=max_trials, tw_penalty=tw_penalty)

        instance_improved_solutions.append(solution2action(improved_solution))
        instance_costs.append(improved_solution.distance() / PYVRP_SCALING_FACTOR)
        instance_feasibility.append(improved_solution.is_feasible())
    return instance_improved_solutions, instance_costs, instance_feasibility


def perform_local_search(
    ls_operator: LocalSearch,
    solution: Solution,
    max_trials: int = 1,
    tw_penalty: int = 2,
) -> Tuple[Solution, bool]:
    
    cost_evaluator = CostEvaluator(
        load_penalty=0,
        tw_penalty=tw_penalty,
        dist_penalty=0,
    )

    current_solution = solution
    best_solution = solution
    best_cost = best_solution.distance()
    for _ in range(max_trials):
        improved_solution = ls_operator(current_solution, cost_evaluator)
        if improved_solution.is_feasible():
            current_solution = improved_solution
            current_cost = current_solution.distance()
            if current_cost <=  best_cost or not best_solution.is_feasible():
                print(f"best cost: { best_cost / PYVRP_SCALING_FACTOR}")
                print(f"Improved solution found with cost: {current_solution.distance() / PYVRP_SCALING_FACTOR}")
                best_solution = current_solution
                best_cost = current_cost
        else:
            # Increase penalties if the solution is infeasible
            tw_penalty *= 10
            cost_evaluator = CostEvaluator(load_penalty=0, tw_penalty=tw_penalty, dist_penalty=0)
    return best_solution


def make_pyvrp_data(td: TensorDict, scaling_factor: int = PYVRP_SCALING_FACTOR):
    num_locs = td["locs"].size(-2)
    for key in td.keys():
        td[key] = scale(td[key], scaling_factor)
    depots = [Depot(x=td["locs"][0, 0], y=td["locs"][0, 1])]
    clients = [
        Client(
            x=td["locs"][i, 0],
            y=td["locs"][i, 1],
            service_duration=td["service_time"][i],
            tw_early=td["time_windows"][..., 0][i],
            tw_late=td["time_windows"][..., 1][i],
        )
        for i in range(1, num_locs)
    ]
    vehicle_types = [
        VehicleType(
            num_available=1,
            tw_early=td["time_windows"][..., 0][0],
            tw_late=td["time_windows"][..., 1][0],
        )
    ]
    return ProblemData(clients, depots, vehicle_types, [td["distance_matrix"]], [td["distance_matrix"]])


def make_search_operator(data: ProblemData, seed=0, neighbourhood_params: Union[dict, None] = None) -> LocalSearch:
    rng = RandomNumberGenerator(seed)
    neighbours = compute_neighbours(data, NeighbourhoodParams(**(neighbourhood_params or {})))
    ls = LocalSearch(data, rng, neighbours)
    #ls.add_node_operator(Exchange10(data))
    for node_operator in NODE_OPERATORS:
        ls.add_node_operator(node_operator(data))
    for route_operator in ROUTE_OPERATORS:
        ls.add_route_operator(route_operator(data))
    return ls


def scale(data: Tensor, scaling_factor: int):
    """
    Scales ands rounds data to integers so PyVRP can handle it.
    """
    array = (data * scaling_factor).numpy().round()
    array = np.where(array == np.inf, np.iinfo(np.int32).max, array)
    array = array.astype(int)

    if array.size == 1:
        return array.item()

    return array


def solution2action(solution: pyvrp.Solution) -> list[int]:
    """
    Converts a PyVRP solution to the action representation, i.e., a giant tour.
    """
    return [visit for route in solution.routes() for visit in route.visits()]
