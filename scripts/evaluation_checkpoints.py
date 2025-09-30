import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
import time
import argparse
import warnings
import logging
import torch
import pandas as pd
from driver.test import test_model_on_random_dataset

# 设置日志级别
logging.getLogger("rl4co").setLevel(logging.ERROR)

# 其他非导入代码
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass

torch.set_float32_matmul_precision("medium")


def evaluate_checkpoints(
    checkpoint_dir,
    env_name="tsptw-lazymask",
    policy="TSPTWPolicy",
    test_path="./data/random/tsptw/test/tsptw49_test_hard_seed2025.npz",
    batch_size=2500,
    seed=2025,
    max_backtrack_steps=600,
    look_ahead_step =2, 
    local_search=False,
    num_workers=6,
    max_trials=3,
    output_csv=None,
):
    """
    Evaluate all model checkpoints in a directory and find the best ones
    based on instance feasibility and solution feasibility rates.

    Args:
        checkpoint_dir: Directory containing model checkpoints
        env_name: Name of the environment (tsptw or tsptw-lazymask)
        test_path: Path to the test dataset
        batch_size: Batch size for inference
        seed: Random seed for reproducibility
        max_backtrack_steps: Maximum number of backtracking steps
        local_search: Whether to use local search
        num_workers: Number of workers for local search
        max_trials: Maximum trials for local search
        output_csv: Path to save metrics CSV (default: metrics_{env_name}_{timestamp}.csv)

    Returns:
        tuple: (best_ins_feas_checkpoint, best_sol_feas_checkpoint)
    """
    if output_csv is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_csv = f"metrics_{env_name}_{timestamp}.csv"
        
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results = []
    best_ins_feas = -float("inf")
    best_sol_feas = -float("inf")
    best_ins_feas_checkpoint = None
    best_sol_feas_checkpoint = None

    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt") or f.endswith(".pth")]

    print(f"Found {len(checkpoint_files)} checkpoints in {checkpoint_dir}")

    for checkpoint_file in sorted(checkpoint_files):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

        print(f"\nEvaluating checkpoint: {checkpoint_file}")

        try:
            # Call the test function
            metrics = test_model_on_random_dataset(
                seed=seed,
                env_name=env_name,
                policy_name=policy,
                checkpoint=checkpoint_path,
                batch_size=batch_size,
                test_path=test_path,
                verbose=False,  # Reduce output verbosity
                max_backtrack_steps=max_backtrack_steps,
                look_ahead_step = look_ahead_step,
                local_search=local_search,
                num_workers=num_workers,
                max_trials=max_trials,
            )
            avg_reward, ins_feas_rate, sol_feas_rate, avg_gap = metrics["avg_reward"], metrics["ins_feas_rate"], metrics["sol_feas_rate"], metrics["avg_gap"]
            # Store results
            result_item = {
                "checkpoint": checkpoint_file,
                "avg_reward": avg_reward.item() if torch.is_tensor(avg_reward) else float(avg_reward),
                "ins_feas_rate": float(ins_feas_rate),
                "sol_feas_rate": float(sol_feas_rate),
                "avg_gap": avg_gap.item() if torch.is_tensor(avg_gap) else float(avg_gap),
            }
            results.append(result_item)

            print(f"Metrics: reward={result_item['avg_reward']:.2f}, ins_feas={result_item['ins_feas_rate']:.4f}, " f"sol_feas={result_item['sol_feas_rate']:.4f}, gap={result_item['avg_gap']:.4f}")

            # Update best checkpoints
            if ins_feas_rate > best_ins_feas:
                best_ins_feas = ins_feas_rate
                best_ins_feas_checkpoint = checkpoint_path
                print(f"  → New best instance feasibility: {ins_feas_rate:.4f}")

            if sol_feas_rate > best_sol_feas:
                best_sol_feas = sol_feas_rate
                best_sol_feas_checkpoint = checkpoint_path
                print(f"  → New best solution feasibility: {sol_feas_rate:.4f}")

            # Save partial results after each evaluation
            pd.DataFrame(results).to_csv(output_csv, index=False)

        except Exception as e:
            print(f"Error evaluating {checkpoint_file}: {str(e)}")

    # Print final results
    print("\n" + "=" * 50)
    print(f"Results saved to {output_csv}")

    if best_ins_feas_checkpoint:
        print(f"\nBest checkpoint (instance feasibility): {os.path.basename(best_ins_feas_checkpoint)}")
        print(f"Instance feasibility rate: {best_ins_feas:.4f}")
    else:
        print("\nNo valid checkpoint found for instance feasibility.")

    if best_sol_feas_checkpoint:
        print(f"\nBest checkpoint (solution feasibility): {os.path.basename(best_sol_feas_checkpoint)}")
        print(f"Solution feasibility rate: {best_sol_feas:.4f}")
    else:
        print("\nNo valid checkpoint found for solution feasibility.")

    return best_ins_feas_checkpoint, best_sol_feas_checkpoint


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, message="The 'warn' method is deprecated, use 'warning' instead")
    warnings.filterwarnings("ignore", message="Attribute.*is an instance of `nn.Module`")

    parser = argparse.ArgumentParser(description="Evaluate model checkpoints and find the best ones")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--env_name", type=str, default="tsptw-lazymask", help="Environment type to use")
    parser.add_argument("--policy", type=str, default="TSPTWPolicy", help="class name of the policy")
    parser.add_argument("--checkpoint_dir", type=str, default="/GLOBALFS/gbu_cinco_2/zhj/checkpoints_repo/tspdl/50-hard/rhoc1-rhon1-1gpu/checkpoints/modified", help="Directory containing model checkpoints")
    parser.add_argument("--output_csv", type=str, default="./results/tspdl50-hard.csv", help="Path to save metrics CSV")
    parser.add_argument("--batch_size", type=int, default=2500)
  
    parser.add_argument("--test_path", type=str, default="./data/random/tsptw/test/tsptw50_test_medium_seed2025.npz")
    parser.add_argument("--max_backtrack_steps", type=int, default=600)
    parser.add_argument("--look_ahead_step", type=int, default=2)
    parser.add_argument("--local_search", action="store_true", help="Use local search during evaluation")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--max_trials", type=int, default=3)

    args = parser.parse_args()

    # Find best checkpoints
    best_ins_feas, best_sol_feas = evaluate_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        env_name=args.env_name,
        policy=args.policy,
        test_path=args.test_path,
        batch_size=args.batch_size,
        seed=args.seed,
        max_backtrack_steps=args.max_backtrack_steps,
        look_ahead_step=args.look_ahead_step,
        local_search=args.local_search,
        num_workers=args.num_workers,
        max_trials=args.max_trials,
        output_csv=args.output_csv,
    )
