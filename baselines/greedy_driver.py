import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

import time
import torch
from rl4co.data.utils import load_npz_to_tensordict
from lmask.utils.data_utils import get_dataloader, extract_info_from_path
from lmask.utils.metric_utils import compute_reward_and_gap_averages
from lmask.utils.utils import seed_everything
from baselines.greedy.tsptw_greedy import TSPTWGreedy, TSPTWLazyMaskGreedy
from baselines.greedy.tspdl_greedy import TSPDLGreedy, TSPDLLazyMaskGreedy


def greedy_solver(
    problem_name,
    test_path,
    verbose=True,
    ref_sol_path=None,
    batch_size=2500,
    seed=2025,
    greedy_type="nearest",
    get_mask=True,
    look_ahead_step=2,
    num_samples = 256,
    max_backtrack_steps = 100,
    env_type = "normal",
):
    """
    Test a model on a random dataset and evaluate its performance

    Args:
        seed: Random seed for reproducibility
        env_name: environment type
        checkpoint: Path to model checkpoint
        batch_size: Batch size for inference
        test_path: Path to test dataset
        verbose: Whether to print detailed information
    Returns:
        tuple: (instance feasibility rate, augmented gap)
    """
    # If ref_sol_path is None, construct it based on problem type
    if ref_sol_path is None:
        test_dir = os.path.dirname(test_path)
        problem_type, problem_size, hardness = extract_info_from_path(test_path)
        reference_solver = "pyvrp" if problem_type == "TSPTW" else "lkh"
        ref_sol_path = os.path.join(test_dir, f"{reference_solver}_{problem_size}_{hardness}.npz")

    print(f"Load test dataset from {test_path}")
    print(f"Load reference solutions from {ref_sol_path}")
    print(f"Test problem: {problem_name}")
    print(f"Greedy Type: {greedy_type}")

    if problem_name == "tsptw":
        if env_type == "normal":
            solver = TSPTWGreedy(greedy_type=greedy_type, get_mask=get_mask, look_ahead_step=look_ahead_step)
        else:
            solver = TSPTWLazyMaskGreedy(max_backtrack_steps=max_backtrack_steps, greedy_type=greedy_type, get_mask=get_mask, look_ahead_step=look_ahead_step)

    elif problem_name == "tspdl":
        if env_type == "normal":
            solver = TSPDLGreedy(greedy_type=greedy_type, get_mask=get_mask, look_ahead_step=look_ahead_step)
        else:
            solver = TSPDLLazyMaskGreedy(max_backtrack_steps=max_backtrack_steps, greedy_type=greedy_type, get_mask=get_mask, look_ahead_step=look_ahead_step)
    else:
        raise ValueError(f"Problem {problem_name} is not supported.")

    # ----------------- Setup -----------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)

    # Load the test data and the reference solutions
    td = load_npz_to_tensordict(test_path)
    dataloader = get_dataloader(td, batch_size=batch_size)
    sol = load_npz_to_tensordict(ref_sol_path)
    cost_bks = sol["costs"].to(device)
    

    # ----------------- Inference -----------------#
    print("Start inference!")
    start = time.time()
    out_list = []
    for batch in dataloader:
        out = solver.rollout(batch, device=device, num_samples=num_samples)
        out_list.append(out)
    inference_time = time.time() - start

    # ----------------- Evaluation -----------------#
    out_td = torch.cat(out_list)

    sol_feas, reward = out_td["sol_feas"], out_td["reward"]  # [B, A, S] or [B, A]
    if sol_feas.dim() == 2:
        ins_feas = sol_feas.any(dim=tuple(range(1, sol_feas.dim())))
    else:
        ins_feas = sol_feas
    ins_feas_rate = ins_feas.float().mean().item()
    sol_feas_rate = sol_feas.float().mean().item()

    masked_reward = reward.masked_fill(~sol_feas, float("-inf"))

    for dim in range(masked_reward.dim() - 1, 0, -1):
        masked_reward = masked_reward.max(dim=dim)[0]

    avg_reward, avg_gap = compute_reward_and_gap_averages(masked_reward, cost_bks)

    if verbose:
        print("=" * 50)
        print(f"Total Inference Time: {inference_time:.2f} s")
        print(f"Instance feasibility rate: {ins_feas_rate:.3%} | Solution feasibility rate: {sol_feas_rate:.3%}")
        print(f"Cost: {-avg_reward:.3f} | Gap: {avg_gap:.3%}")

    return {"inference_time": inference_time, "ins_feas_rate": ins_feas_rate, "sol_feas_rate": sol_feas_rate, "avg_reward": avg_reward, "avg_gap": avg_gap}


if __name__ == "__main__":
    import pandas as pd
    import os
    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", type=str, default="lazymask", choices=["normal", "lazymask"], help="Type of environment to use")
    parser.add_argument("--max_backtrack_steps", type=int, default=200, help="Maximum backtrack steps for lazy mask environment")
    parser.add_argument("--look_ahead_step", type=int, default=1, help="Look ahead steps for greedy policy")
    parser.add_argument("--sample_size", type=int, default=128, help="Number of samples for greedy policy")
    args = parser.parse_args()
    
    # Set default parameters
    data_dir = "data/random"
    look_ahead_step = args.look_ahead_step
    seed = 2025
    sample_size = args.sample_size
    # Define parameter combinations to test
    problems = ["tsptw", "tspdl"]
    problem_sizes = [50, 100]
    hardness_levels = ["easy", "medium", "hard"]
    greedy_types = ["random_nearest", "random_min_resource"]

    # Initialize results list
    results = []

    # Create result directory if it doesn't exist
    result_dir = os.path.abspath(os.path.join(current_dir, os.pardir, "result"))
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    result_file = os.path.join(result_dir, "greedy_results.csv")

    print(f"Starting greedy solver evaluation across all parameter combinations")
    print(f"Results will be saved to {result_file}")

    # Iterate through all combinations
    for problem in problems:
        for problem_size in problem_sizes:
            for hardness in hardness_levels:
                for greedy_type in greedy_types:
                    print(f"\n{'='*80}")
                    print(f"Testing: Problem={problem}, Size={problem_size}, Hardness={hardness}, Greedy Type={greedy_type}")
                    print(f"{'='*80}")

                    # Construct test path and reference solution path
                    test_path = f"{data_dir}/{problem}/test/{problem}{problem_size}_test_{hardness}_seed{seed}.npz"
                    test_dir = os.path.dirname(test_path)
                    reference_solver = "pyvrp" if problem == "tsptw" else "lkh"
                    ref_sol_path = os.path.join(test_dir, f"{reference_solver}_{problem_size}_{hardness}.npz")

                    # Check if test file exists
                    if not os.path.exists(test_path):
                        print(f"Warning: Test file {test_path} does not exist, skipping...")
                        continue

                    if not os.path.exists(ref_sol_path):
                        print(f"Warning: Reference solution file {ref_sol_path} does not exist, skipping...")
                        continue
                    batch_size = 1024 if problem_size == 50 else 512
                    # Run the greedy solver
                    result = greedy_solver(
                        problem_name=problem, test_path=test_path, ref_sol_path=ref_sol_path, batch_size=batch_size, greedy_type=greedy_type, look_ahead_step=look_ahead_step, seed=seed, verbose=True, num_samples = sample_size, 
                        env_type=args.env_type, max_backtrack_steps=args.max_backtrack_steps
                        
                    )

                    # Format the results with percentage symbols using f-strings
                    ins_infeas_rate = (1 - result["ins_feas_rate"]) * 100
                    sol_infeas_rate = (1 - result["sol_feas_rate"]) * 100
                    gap = result["avg_gap"] * 100
                    cost = -result["avg_reward"]

                    result_entry = {
                        "problem": problem,
                        "problem_size": problem_size,
                        "hardness": hardness,
                        "greedy_type": greedy_type,
                        "inference_time": int(result["inference_time"]),
                        "cost": f"{cost:.2f}",
                        "ins_infeas_rate": f"{ins_infeas_rate:.2f}%",
                        "sol_infeas_rate": f"{sol_infeas_rate:.2f}%",
                        "gap": f"{gap:.2f}%",
                    }

                    results.append(result_entry)

                    # Save intermediate results after each test
                    pd.DataFrame(results).to_csv(result_file, index=False)
                    print(f"Results updated in {result_file}")

    print(f"All tests completed! Final results saved to {result_file}")
