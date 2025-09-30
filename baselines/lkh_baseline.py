import os
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, os.pardir))
sys.path.append(project_root)
import time
import subprocess
import numpy as np
import torch
from tensordict.tensordict import TensorDict

from baselines.utils import scale
LKH_SCALING_FACTOR = 1000


def lkh_solve(
    td: TensorDict,
    problem_type: str,
    problem_name: str,
    save_dir: str = "lkh_results",
    executable: str = "./LKH",
    **kwargs,
):
    paths = create_file_paths(save_dir, problem_name)

    make_vrplib_file(td, problem_type, paths["problem_file"])
    kwargs["PROBLEM_FILE"] = paths["problem_file"]
    kwargs["TOUR_FILE"] = paths["tour_file"]
    make_parameter_file(paths["parameter_file"], **kwargs)

    try:
        start_time = time.perf_counter()
        with open(f"{save_dir}/lkh_log.txt", "a") as f:
            f.write(f"Running LKH for problem {problem_name}\n")
            subprocess.check_call([executable, paths["parameter_file"]], stdout=f, stderr=f)
            f.write(f"Completed LKH for problem {problem_name}\n")
        duration = time.perf_counter() - start_time
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running LKH: {e}")
        duration = np.inf

    if os.path.isfile(paths["tour_file"]):
        tour, length = read_tour_file(paths["tour_file"])
        return tour, length, duration
    else:
        print(f"LKH3 fails to find a feasible solution for Instance {problem_name}")
        return np.arange(td["locs"].size(-2)), np.inf, duration


def create_file_paths(base_dir, problem_name):
    sub_dirs = ["Instances", "Tours", "Paramas"]
    paths = {}
    for sub_dir in sub_dirs:
        full_dir = os.path.join(base_dir, sub_dir)
        os.makedirs(full_dir, exist_ok=True)
        if sub_dir == "Instances":
            paths["problem_file"] = os.path.join(full_dir, f"{problem_name}.tsptw")
        elif sub_dir == "Tours":
            paths["tour_file"] = os.path.join(full_dir, f"{problem_name}.tour")
        elif sub_dir == "Paramas":
            paths["parameter_file"] = os.path.join(full_dir, f"{problem_name}.par")
    return paths


def _is_1D(data) -> bool:
    for elt in data:
        if isinstance(elt, (list, tuple, np.ndarray)):
            return False
    return True


def _format(name: str, data) -> str:
    section = [name]
    include_idx = name not in ["EDGE_WEIGHT_SECTION"]
    if _is_1D(data):
        for idx, elt in enumerate(data, 1):
            prefix = f"{idx}\t" if include_idx else ""
            section.append(prefix + str(elt))
    else:
        for idx, row in enumerate(data, 1):
            prefix = f"{idx}\t" if include_idx else ""
            rest = "\t".join([str(elt) for elt in row])
            section.append(prefix + rest)

    return "\n".join(section)


def make_vrplib_file(td: TensorDict, problem_type: str, file_path: str, scaling_factor: int = LKH_SCALING_FACTOR):
    num_locs = td["locs"].size(-2)

    # Data speicifications
    specs = {
        "TYPE": problem_type,
        "DIMENSION": num_locs,
        "EDGE_WEIGHT_TYPE": "EXPLICIT",
        "EDGE_WEIGHT_FORMAT": "FULL_MATRIX",
    }

    # Data section
    sections = {
        "EDGE_WEIGHT_SECTION": scale(td["distance_matrix"], scaling_factor),
    }
    if problem_type == "TSPTW":
        sections["TIME_WINDOW_SECTION"] = scale(td["time_windows"], scaling_factor)
    elif problem_type == "TSPDL":
        sections["DEMAND_SECTION"] = scale(td["demand"], scaling_factor)
        sections["DRAFT_LIMIT_SECTION"] = scale(td["draft_limit"], scaling_factor)

    problem = "\n".join([f"{k} : {v}" for k, v in specs.items()])
    problem += "\n" + "\n".join([_format(name, data) for name, data in sections.items()])
    problem += "\n" + "\n".join(["DEPOT_SECTION", "1", "-1", "EOF"])
    with open(file_path, "w") as f:
        f.write(problem)


def make_parameter_file(file_path, **kwargs):
    with open(file_path, "w") as file:
        file.write("SPECIAL\n")
        for key, value in kwargs.items():
            file.write(f"{key} = {value}\n")


def read_tour_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    length = int([line for line in lines if line.startswith("COMMENT : Length =")][0].split("=")[1].strip())
    tour_start = lines.index("TOUR_SECTION\n") + 1
    tour_end = lines.index("-1\n")
    tour = [int(line.strip()) - 1 for line in lines[tour_start:tour_end]]

    return tour, length / LKH_SCALING_FACTOR


if __name__ == "__main__":
    from rl4co.data.utils import load_npz_to_tensordict
    td = load_npz_to_tensordict("./data/random/tsptw/test/tsptw49_test_hard_seed2025.npz")
    distance_matrix = torch.cdist(td["locs"], td["locs"], p=2)
    distance_matrix.diagonal(dim1=-2, dim2=-1).fill_(0)
    td.update({"distance_matrix": distance_matrix})
    td_test = td[0]
    tour, length, duration = lkh_solve(td_test, problem_type="TSPTW", problem_name="random1")
    print(f"After {duration:.2f} seconds, LKH3 finds a tour of length {length:.2f} for the test instance. The tour is \n{tour}")
