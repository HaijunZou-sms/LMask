import os
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, os.pardir))
sys.path.append(project_root)
import time
import torch
from pyvrp import Client, Depot, ProblemData, VehicleType, solve as _solve
from pyvrp.stop import MaxRuntime
from tensordict.tensordict import TensorDict
from rl4co.data.utils import load_npz_to_tensordict
from baselines.utils import scale

PYVRP_SCALING_FACTOR = 1000


def pyvrp_solve(td: TensorDict, max_runtime: float = 20):
    data = make_pyvrp_data(td)
    start_time = time.perf_counter()
    result = _solve(data, stop=MaxRuntime(max_runtime), display=True)
    duration = time.perf_counter() - start_time
    solution = pyvrp_solution2action(result.best)
    cost = result.cost() / PYVRP_SCALING_FACTOR 
    return solution, cost, duration


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
            tw_early=td["time_windows"][i][0],
            tw_late=td["time_windows"][i][1],
        )
        for i in range(1, num_locs)
    ]
    vehicle_types = [
        VehicleType(num_available=1, tw_early=td["time_windows"][0, 0], tw_late=td["time_windows"][0, 1])
    ]
    return ProblemData(
        clients, depots, vehicle_types, [td["distance_matrix"]], [td["distance_matrix"]]
    )


def pyvrp_solution2action(solution):
    return [0] + [visit for route in solution.routes() for visit in route.visits()]


if __name__ == "__main__":
    # Example usage
    file_path = "../data/random/tsptw/test/tsptw50_test_medium_seed2025.npz"
    td = load_npz_to_tensordict(file_path)
    td = td[0]
    distance_matrix = torch.cdist(td["locs"], td["locs"], p=2)
    distance_matrix.diagonal(dim1=-2, dim2=-1).fill_(0)
    td.update({"distance_matrix": distance_matrix})
    solution, cost, duration = pyvrp_solve(td, max_runtime=360)
    print(f"Solution: {solution}, Cost: {cost}, Duration: {duration}")