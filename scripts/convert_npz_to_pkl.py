import os
import pickle
import numpy as np
from tensordict import TensorDict
import torch
import glob


def load_npz_to_tensordict(path):
    """Load npz file and convert to TensorDict"""
    data = dict(np.load(path))
    tensor_dict = {}

    # Convert numpy arrays to PyTorch tensors
    for k, v in data.items():
        tensor_dict[k] = torch.from_numpy(v)

    return TensorDict(tensor_dict, batch_size=tensor_dict[list(tensor_dict.keys())[0]].shape[0])


def convert_tsptw_to_pkl_format(tensordict):
    """Convert TSPTW TensorDict to pkl format"""
    batch_size = tensordict.batch_size[0]
    result = []

    for i in range(batch_size):
        # Extract data for this instance
        locs = tensordict["locs"][i].numpy().tolist()
        service_time = tensordict["service_time"][i].numpy().tolist()
        time_windows = tensordict["time_windows"][i].numpy()

        # Split time windows into early and late
        tw_early = time_windows[:, 0].tolist()
        tw_late = time_windows[:, 1].tolist()

        # Create tuple for this instance
        instance = (locs, service_time, tw_early, tw_late)
        result.append(instance)

    return result


def convert_tspdl_to_pkl_format(tensordict):
    """Convert TSPDL TensorDict to pkl format"""
    batch_size = tensordict.batch_size[0]
    result = []

    for i in range(batch_size):
        # Extract data for this instance
        locs = tensordict["locs"][i].numpy().tolist()
        demand = tensordict["demand"][i].numpy().tolist()
        draft_limit = tensordict["draft_limit"][i].numpy().tolist()

        # Create tuple for this instance
        instance = (locs, demand, draft_limit)
        result.append(instance)

    return result


def convert_ref_sol_npz_to_pkl(npz_path, problem_type=None):
    # Create output directory
    output_dir = os.path.join(os.path.dirname(npz_path), "pkl")
    os.makedirs(output_dir, exist_ok=True)

    # Output path
    output_path = os.path.join(output_dir, os.path.basename(npz_path).replace(".npz", ".pkl"))

    # Load and convert
    print(f"Converting {npz_path} to {output_path}")

    sol = load_npz_to_tensordict(npz_path)
    if problem_type.lower() == "tsptw":
        data = (sol["costs"].tolist() * 100, sol["solutions"].tolist())
    elif problem_type.lower() == "tspdl":
        data = (sol["costs"].tolist(), sol["solutions"].tolist())
    dataset = list(zip(*data))
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    return output_path


def process_solver_solutions(data_dir):
    """
    Process all solution files starting with 'pyvrp' or 'lkh' in the given directory.
    Extract problem type from the filepath and convert NPZ files to PKL using convert_ref_sol_npz_to_pkl.

    Args:
        data_dir: Directory containing solver solution files
    """
    # Find all NPZ files that start with pyvrp or lkh
    pattern = os.path.join(data_dir, "**", "*.npz")
    all_npz_files = glob.glob(pattern, recursive=True)

    # Filter for files that start with pyvrp or lkh
    solver_files = [f for f in all_npz_files if os.path.basename(f).startswith(("pyvrp", "lkh"))]

    if not solver_files:
        print(f"No pyvrp or lkh solution files found in {data_dir}")
        return

    print(f"Found {len(solver_files)} solver solution files to convert")

    for npz_path in solver_files:
        # Determine problem type from path
        if "tsptw" in npz_path.lower():
            problem_type = "tsptw"
        elif "tspdl" in npz_path.lower():
            problem_type = "tspdl"
        else:
            print(f"Skipping {npz_path}: Unable to determine problem type")
            continue

        # Convert the file
        try:
            output_path = convert_ref_sol_npz_to_pkl(npz_path, problem_type=problem_type)
            print(f"Successfully converted to {output_path}")
        except Exception as e:
            print(f"Error processing {npz_path}: {e}")


def convert_npz_to_pkl(npz_path, problem_type=None):
    """Convert a single npz file to pkl format"""
    # Determine problem type from path if not specified
    if problem_type is None:
        if "tsptw" in npz_path.lower():
            problem_type = "tsptw"
        elif "tspdl" in npz_path.lower():
            problem_type = "tspdl"
        else:
            raise ValueError(f"Could not determine problem type from path: {npz_path}")

    # Create output directory
    output_dir = os.path.join(os.path.dirname(npz_path), "pkl")
    os.makedirs(output_dir, exist_ok=True)

    # Output path
    output_path = os.path.join(output_dir, os.path.basename(npz_path).replace(".npz", ".pkl"))

    # Load and convert
    print(f"Converting {npz_path} to {output_path}")
    tensordict = load_npz_to_tensordict(npz_path)

    if problem_type == "tsptw":
        result = convert_tsptw_to_pkl_format(tensordict)
    elif problem_type == "tspdl":
        result = convert_tspdl_to_pkl_format(tensordict)
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    return output_path


def convert_all_npz_files(base_dir):
    """Convert all npz files in a directory structure to pkl"""
    # Find all npz files
    npz_files = glob.glob(os.path.join(base_dir, "**", "*.npz"), recursive=True)

    # Group by problem type
    tsptw_files = [f for f in npz_files if "tsptw" in os.path.basename(f).lower()]
    tspdl_files = [f for f in npz_files if "tspdl" in os.path.basename(f).lower()]

    print(f"Found {len(tsptw_files)} TSPTW files and {len(tspdl_files)} TSPDL files")

    # Convert all files
    for npz_path in tsptw_files:
        convert_npz_to_pkl(npz_path, problem_type="tsptw")

    for npz_path in tspdl_files:
        convert_npz_to_pkl(npz_path, problem_type="tspdl")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/random/tspdl")
    args = parser.parse_args()
    # convert_all_npz_files(args.data_dir)
    process_solver_solutions(args.data_dir)
    print("Conversion completed!")
