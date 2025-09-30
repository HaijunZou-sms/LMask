import os
import numpy as np
import torch
from lmask.utils.utils import seed_everything


def check_extension(filename, extension=".npz"):
    """Check that filename has extension, otherwise add it"""
    if os.path.splitext(filename)[1] != extension:
        return filename + extension
    return filename

def generate_draft_limits(dataset_size, num_loc, hardness="hard", draft_method="rejection"):
    constrain_pct = {"hard": 0.9, "medium": 0.75, "easy": 0.5}[hardness]
    num_constrained = int(constrain_pct * (num_loc + 1))
    total_load = num_loc
    draft_limit = np.full((dataset_size, num_loc + 1), total_load, dtype=np.float32)

    if draft_method == "rejection":
        for i in range(dataset_size):
            selected_ports = np.random.choice(range(1, num_loc + 1), num_constrained, replace=False)
            feasible = False
            while not feasible:
                constrained_limits = np.random.randint(1, total_load, size=num_constrained)
                cnt = np.bincount(constrained_limits, minlength=total_load + 1)
                cum_counts = np.cumsum(cnt)
                feasible = np.all(cum_counts <= np.arange(len(cum_counts)))
            draft_limit[i, selected_ports] = constrained_limits.astype(np.float32)

    elif draft_method == "clamp":
        selected_ports = np.argsort(-np.random.rand(dataset_size, num_loc), axis=1)[:, :num_constrained] + 1

        constrained_values = np.random.randint(1, total_load + 1, (dataset_size, num_constrained))
        sorted_vals = -np.sort(-constrained_values, axis=1)  # sort in descending order

        positions = np.arange(1, num_constrained + 1)
        min_required = num_loc - positions + 1
        clamped_vals = np.maximum(sorted_vals, min_required[None, :])

        batch_indices = np.arange(dataset_size)[:, None]
        draft_limit[batch_indices, selected_ports] = clamped_vals.astype(np.float32)

    return draft_limit


def generate_tspdl_data(
    dataset_size=10_000,
    num_loc=50,
    min_loc=0.0,
    max_loc=1.0,
    hardness="hard",
    draft_method="rejection",
    **kwargs,
):
    print(f"Generating {dataset_size} instances of TSPDL with {num_loc} locations and {hardness} hardness level")

    locs = np.random.uniform(size=(dataset_size, num_loc + 1, 2)) * (max_loc - min_loc) + min_loc

    demand = np.zeros((dataset_size, num_loc + 1), dtype=np.float32)
    demand[:, 1:] = 1.0

    draft_limit = generate_draft_limits(dataset_size, num_loc, hardness, draft_method)

    total_load = num_loc
    locs = (locs - min_loc) / (max_loc - min_loc)
    demand = demand / total_load
    draft_limit = draft_limit / total_load

    return {
        "locs": locs.astype(np.float32),
        "demand": demand.astype(np.float32),
        "draft_limit": draft_limit.astype(np.float32),
    }


def generate_random_time_windows(batch_size, num_loc, expected_distance, alpha, beta):
    tw_early = np.random.randint(0, expected_distance, (batch_size, num_loc + 1))
    tw_early[:, 0] = 0

    epsilon = np.random.uniform(alpha, beta, (batch_size, num_loc + 1))
    tw_width = np.round(epsilon * expected_distance)
    tw_width[:, 0] = 2 * expected_distance

    tw_late = tw_early + tw_width

    return np.stack([tw_early, tw_late], axis=-1, dtype=np.float64)


def generate_time_windows_from_randomperm(batch_size, locs, randperm, max_tw_width):
    # Get permuted locations and calculate windows
    num_loc = locs.shape[1] - 1
    idx = randperm[..., None].repeat(2, axis=2)
    locs_perm = np.take_along_axis(locs, idx, axis=1)

    arrival_time = np.cumsum(np.linalg.norm(locs_perm[:, :-1] - locs_perm[:, 1:], axis=-1), axis=-1)

    # Generate windows in permuted order
    tw_half_width = np.random.uniform(0, max_tw_width / 2, (2, batch_size, num_loc))
    tw_perm = np.zeros((batch_size, num_loc + 1, 2))
    tw_perm[:, 1:, 0] = np.maximum(arrival_time - tw_half_width[0], 0)
    tw_perm[:, 1:, 1] = arrival_time + tw_half_width[1]
    tw = np.zeros_like(tw_perm)
    np.put_along_axis(tw, idx, tw_perm, 1)

    return tw


def generate_tsptw_data(
    dataset_size=10_000,
    num_loc=50,
    min_loc=0.0,
    max_loc=100.0,
    hardness="hard",
    **kwargs,
):
    print(f"Generating {dataset_size} instances of TSPTW with {num_loc} locations and {hardness} hardness level ")
    locs = np.random.uniform(size=(dataset_size, num_loc + 1, 2)) * (max_loc - min_loc)

    if hardness == "easy":
        tw = generate_random_time_windows(
            batch_size=dataset_size,
            num_loc=num_loc,
            expected_distance=55 * (num_loc + 1),
            alpha=0.5,
            beta=0.75,
        )

    elif hardness == "medium":
        tw = generate_random_time_windows(
            batch_size=dataset_size,
            num_loc=num_loc,
            expected_distance=55 * (num_loc + 1),
            alpha=0.1,
            beta=0.2,
        )

    elif hardness == "hard":
        max_tw_width = kwargs.pop("max_tw_width", 100.0)
        randperm = np.concatenate(
            (
                np.zeros((dataset_size, 1), dtype=int),
                1 + np.array([np.random.permutation(num_loc) for _ in range(dataset_size)]),
            ),
            axis=1,
        )

        tw = generate_time_windows_from_randomperm(dataset_size, locs, randperm, max_tw_width)

    # Normalize
    loc_scaler = max_loc - min_loc
    locs, tw = locs / loc_scaler, tw / loc_scaler
    d_i0 = np.linalg.norm(locs[:, 1:] - locs[:, 0:1], axis=-1)
    tw[:, 0, 1] = np.max(tw[:, 1:, 1] + d_i0, axis=-1)

    return {
        "locs": locs.astype(np.float32),
        "service_time": np.zeros((dataset_size, num_loc + 1), dtype=np.float32),
        "time_windows": tw.astype(np.float32),
    }


def generate_dataset(
    filename=None,
    data_dir="data/random",
    problem="tsptw",
    dataset_size=10_000,
    num_locs=[49],
    seed=2025,
    **kwargs,
):
    """We keep a similar structure as in Kool et al. 2019 but save and load the data as npz
    This is way faster and more memory efficient than pickle and also allows for easy transfer to TensorDict
    """

    hardness = kwargs.get("hardness", "hard")
    fname = filename
    if isinstance(num_locs, int):
        num_locs = [num_locs]
    for num_loc in num_locs:
        datadir = os.path.join(data_dir, problem)
        os.makedirs(datadir, exist_ok=True)

        if filename is None:
            fname = os.path.join(data_dir, f"{problem}{num_loc}_{hardness}_seed{seed}.npz")
        else:
            fname = check_extension(filename, extension=".npz")

        # Generate any needed directories
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        # Set seed
        seed_everything(seed=seed)

        # Automatically generate dataset
        if problem == "tsptw":
            dataset = generate_tsptw_data(dataset_size=dataset_size, num_loc=num_loc, **kwargs)
        elif problem == "tspdl":
            dataset = generate_tspdl_data(dataset_size=dataset_size, num_loc=num_loc, **kwargs)

        # Save to disk as dict
        print("Saving {} data to {}".format(problem.upper(), fname))
        np.savez(fname, **dataset)


if __name__ == "__main__":
    data_dir = "data/random"
    seeds = {
        "val": 4321,
        "test": 2025,
    }
    sizes = [49, 99]
    hard_levels = ["easy", "medium", "hard"]
    problems = ["TSPTW", "TSPDL"]
    dataset_sizes = {
        "val": 256,
        "test": 10_000,
    }

    for problem in problems:
        problem = problem.lower()
        for phase, seed in seeds.items():
            for size in sizes:
                # For TSPDL, only generate medium and hard difficulty levels
                problem_hard_levels = hard_levels if problem == "tsptw" else ["medium", "hard"]
                for hardness in problem_hard_levels:
                    generate_dataset(
                        problem=problem,
                        data_dir=data_dir,
                        filename=data_dir + f"/{problem}/{phase}/{problem}{size+1}_{phase}_{hardness}_seed{seed}.npz",
                        dataset_size=dataset_sizes[phase],
                        num_locs=size,
                        seed=seed,
                        hardness=hardness,
                    )
