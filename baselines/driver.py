import os
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, os.pardir))
sys.path.append(project_root)
import argparse
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import numpy as np
import torch
from rl4co.data.utils import load_npz_to_tensordict
import baselines.pyvrp_baseline as pyvrp
import baselines.lkh_baseline as lkh
import baselines.ortools_baseline as ortools
from lmask.utils.data_utils import extract_info_from_path


def load_npz_data(file_path="./data/random/tsptw/test/tsptw50_test_hard_seed2025.npz"):
    td = load_npz_to_tensordict(file_path)
    distance_matrix = torch.cdist(td["locs"], td["locs"], p=2)
    distance_matrix.diagonal(dim1=-2, dim2=-1).fill_(0)
    td.update({"distance_matrix": distance_matrix})
    return td


def wrapper(args, func):
    """
    A wrapper function designed to work with multiprocessing.Pool.imap.

    pool.imap requires the function it calls to accept a single argument.
    However, our solver functions need multiple arguments. This wrapper
    allows us to:
    1. Pack multiple arguments into a single tuple (args)
    2. Pass that tuple to pool.imap
    3. Unpack the tuple back into separate arguments using *args
    4. Call the actual function (func) with those unpacked arguments

    Args:
        args: A tuple containing all arguments to be unpacked for the function
        func: The actual function to call with the unpacked arguments

    Returns:
        The result of calling func with the unpacked arguments
    """
    return func(*args)


def solve(file_path, solver: str = "pyvrp", num_procs: int = 1, **kwargs):
    instances = load_npz_data(file_path)
    solvers = {"pyvrp": pyvrp.pyvrp_solve, "lkh": lkh.lkh_solve, "ortools": ortools.ortools_solve}
    func = partial(solvers[solver], **kwargs)
    problem_type, problem_size, hard_level = extract_info_from_path(file_path)

    if solver == "lkh":
        save_dir = f"results/{problem_type.lower()}/lkh_{problem_size}_{hard_level}"
        args = [(instance, problem_type, f"problem_{i:04d}", save_dir) for i, instance in enumerate(instances)]
        wrapped_func = partial(wrapper, func=func)
    else:
        args = instances
        wrapped_func = func

    if num_procs > 1:
        with Pool(processes=num_procs) as pool:
            results = list(tqdm(pool.imap(wrapped_func, args), total=len(args)))
    else:
        results = [wrapped_func(arg) for arg in tqdm(args)]

    solutions, costs, durations = zip(*results)
    solutions, costs, durations = np.array(solutions), np.array(costs), np.array(durations)
    valid_costs = costs[np.isfinite(costs)]
    avg_cost = np.mean(valid_costs) if len(valid_costs) > 0 else float("inf")

    print(f"Instance feasiblity rate: {np.mean(np.isfinite(costs)):.2%}")
    print(f"Average cost: {avg_cost:.2f}")
    print(f"Total serial duration: {np.sum(durations):.2f}")
    print(f"Estimated parallel duration: {np.sum(durations)/num_procs:.2f}")
    np.savez(
        f"{solver}_{problem_size}_{hard_level}.npz",
        solutions=solutions,
        costs=costs,
        durations=durations,
    )

    return solutions, costs, durations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve VRP instances using different solvers.")
    parser.add_argument("--file_path", type=str, default="../data/random/tsptw/tsptw100_hard.npz", help="Path to the data file.")
    parser.add_argument("--solver", type=str, choices=["pyvrp", "lkh", "ortools"], default="pyvrp", help="Solver to use.")
    parser.add_argument("--num_procs", type=int, default=1, help="Number of processes to use.")
    parser.add_argument("--max_runtime", type=float, default=20, help="Maximum runtime for pyvrp solver.")
    parser.add_argument("--executable", type=str, default="./LKH", help="Path to the LKH executable.")
    parser.add_argument("--max_trials", type=int, default=10000, help="Maximum trials for LKH solver.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for LKH solver.")
    parser.add_argument("--trace_level", type=int, default=1, help="Trace level for LKH solver.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for LKH solver.")
    args = parser.parse_args()

    if args.solver == "pyvrp" or args.solver == "ortools":
        solve(
            args.file_path,
            solver=args.solver,
            num_procs=args.num_procs,
            max_runtime=args.max_runtime,
        )
    elif args.solver == "lkh":
        solve(
            args.file_path,
            solver=args.solver,
            num_procs=args.num_procs,
            executable=args.executable,
            MAX_TRIALS=args.max_trials,
            RUNS=args.runs,
            TRACE_LEVEL=args.trace_level,
            SEED=args.seed,
        )
    
