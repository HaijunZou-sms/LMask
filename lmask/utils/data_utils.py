import os
import torch
from tensordict.tensordict import TensorDict
from torch.utils.data import DataLoader
from rl4co.data.dataset import TensorDictDataset


def get_dataloader(td, batch_size=4):
    """Get a dataloader from a TensorDictDataset"""
    # Set up the dataloader
    dataloader = DataLoader(
        TensorDictDataset(td.clone()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=TensorDictDataset.collate_fn,
    )
    return dataloader


def extract_info_from_path(file_path):
    """
    Extract problem information from file path.

    Example paths:
    - data/random/tsptw/test/tsptw50_test_hard_seed2025.npz
    - data/random/tspdl/test/tspdl100_test_hard_seed2025.npz

    Returns:
        tuple: (problem_type, problem_size, hard_level)
    """
    file_name = os.path.basename(file_path)
    # Extract problem type (TSPTW or TSPDL)
    if "tsptw" in file_name.lower():
        problem_type = "TSPTW"
    elif "tspdl" in file_name.lower():
        problem_type = "TSPDL"
    else:
        raise ValueError(f"Unknown problem type in filename: {file_name}")
    problem_size = file_name.split(problem_type.lower())[1].split("_")[0]
    hardness_level = file_name.split("_")[2]
    return problem_type, problem_size, hardness_level


def read_tsptw_instance(file_path: str) -> dict:
    """
    Read TSPTW instance in the da Silva-Urrutia format, return a dictionary containing the following fields:
    - locs: a tensor of shape (n+1, 2) containing the coordinates of n locations
    - service_time: a tensor of shape (n+1,) containing the service time of each location
    - time_windows: a tensor of shape (n+1, 2) containing the time windows of each location
    """
    locs = []
    service_time = []
    time_windows = []

    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines[6:-1]:
            parts = line.split()
            if len(parts) == 7:
                x_coord = float(parts[1])
                y_coord = float(parts[2])
                locs.append([x_coord, y_coord])
                service_time.append(float(parts[6]))
                time_windows.append([float(parts[4]), float(parts[5])])
    td = TensorDict(
        {
            "locs": torch.tensor(locs, dtype=torch.float32).unsqueeze(0),
            "service_time": torch.tensor(service_time, dtype=torch.float32).unsqueeze(0),
            "time_windows": torch.tensor(time_windows, dtype=torch.float32).unsqueeze(0),
        },
        batch_size=[1],
    )
    return td
