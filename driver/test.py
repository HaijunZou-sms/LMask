import time
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
import argparse
import warnings

warnings.filterwarnings("ignore", message="Unused keyword arguments:.*")
import logging
import torch
from rl4co.data.utils import load_npz_to_tensordict

from lmask.envs import get_env
import lmask.models.policy
from lmask.utils.data_utils import get_dataloader, extract_info_from_path
from lmask.utils.metric_utils import compute_reward_and_gap_averages
from lmask.utils.utils import infer_default_cofigs, seed_everything


logging.getLogger("rl4co").setLevel(logging.ERROR)

try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass

torch.set_float32_matmul_precision("medium")


def test_model_on_random_dataset(
    env_name,
    policy_name,
    test_path,
    checkpoint,
    verbose=True,
    ref_sol_path=None,
    batch_size=2500,
    seed=2025,
    use_reld = False,
    **kwargs,
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
    print(f"Load policy from {checkpoint}")
    print(f"Use environment {env_name}")
    # ----------------- Setup -----------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)

    rollout_kwargs = {k: kwargs.pop(k) for k in ["decode_type", "num_samples"] if k in kwargs}

    # Instantiate the environment
    env = get_env(env_name, **kwargs)

    # Load the policy
    policy_class = getattr(lmask.models.policy, policy_name)
    policy = policy_class(decoder_class="reld") if use_reld else policy_class()
    policy.load_state_dict(torch.load(checkpoint))
    policy.to(device).eval()

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
        out = env.rollout(batch, policy, **rollout_kwargs)
        out_list.append(out)
    inference_time = time.time() - start

    # ----------------- Evaluation -----------------#
    out_td = torch.cat(out_list)

    sol_feas, reward = out_td["sol_feas"], out_td["reward"]  # [B, A, S] or [B, A]
    ins_feas = sol_feas.any(dim=tuple(range(1, sol_feas.dim())))
    ins_feas_rate = ins_feas.float().mean().item()
    sol_feas_rate = sol_feas.float().mean().item()

    masked_reward = reward.masked_fill(~sol_feas, float("-inf"))
    no_aug_masked = masked_reward[:, 0]
    for dim in range(masked_reward.dim() - 1, 0, -1):
        masked_reward = masked_reward.max(dim=dim)[0]
    for dim in range(no_aug_masked.dim() - 1, 0, -1):
        no_aug_masked = no_aug_masked.max(dim=dim)[0]

    avg_reward, avg_gap = compute_reward_and_gap_averages(masked_reward, cost_bks)
    avg_no_aug_reward, avg_no_aug_gap = compute_reward_and_gap_averages(no_aug_masked, cost_bks)

    if verbose:
        print("=" * 50)
        print(f"Total Inference Time: {inference_time:.2f} s")
        print(f"Instance feasibility rate: {ins_feas_rate:.3%} | Solution feasibility rate: {sol_feas_rate:.3%}")
        print(f"No augment| Cost: {-avg_no_aug_reward:.3f} | Gap: {avg_no_aug_gap:.3%}")
        print(f"Augmented | Cost: {-avg_reward:.3f} | Gap: {avg_gap:.3%}")

    return {"inference_time": inference_time, "ins_feas_rate": ins_feas_rate, "sol_feas_rate": sol_feas_rate, "avg_reward": avg_reward, "avg_gap": avg_gap}


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="Attribute.*is an instance of `nn.Module`")

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch_size", type=int, default=2500)

    # Parameters fort test
    parser.add_argument("--problem", type=str, choices=["tspdl", "tsptw"], default="tsptw", help="Problem type")
    parser.add_argument("--problem_size", type=int, default=50, help="Problem size")
    parser.add_argument("--hardness", type=str, choices=["easy", "medium", "hard"], default="hard", help="Problem difficulty")

    # Optional test parameters (can be inferred)
    parser.add_argument("--env_name", type=str, help="Environment name")
    parser.add_argument("--policy_name", type=str, help="Class name of the policy")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--test_path", type=str, help="Path to test dataset")
    parser.add_argument("--ref_sol_path", type=str, help="Path to reference solutions")

    # Algorithm parameters
    parser.add_argument("--lookahead_step", "-L", type=int, choices=[1, 2, 3], default=2, help="Number of lookahead steps when getting inital masks")
    parser.add_argument("--max_backtrack_steps", "-R", type=int, default=100)
    parser.add_argument("--decode_type", type=str, default="greedy", choices=["greedy", "sampling"], help="Decoding strategy")
    parser.add_argument("--num_samples", "-N", type=int, default=1)

    args = parser.parse_args()

    # Infer parameters if needed
    if args.problem and args.problem_size and args.hardness:
        inferred = infer_default_cofigs(args.problem, args.problem_size, args.hardness, args.seed)

        # Only use inferred values for parameters that weren't explicitly provided
        if not args.policy_name:
            args.policy_name = inferred["policy_name"]
        if not args.checkpoint:
            args.checkpoint = inferred["checkpoint"]
        if not args.test_path:
            args.test_path = inferred["test_path"]
        if not args.env_name:
            args.env_name = inferred["env_name"]

    metric_dict = test_model_on_random_dataset(
        seed=args.seed,
        env_name=args.env_name,
        policy_name=args.policy_name,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        test_path=args.test_path,
        verbose=True,
        max_backtrack_steps=args.max_backtrack_steps,
        lookahead_step=args.lookahead_step,
        decode_type=args.decode_type,
        num_samples=args.num_samples,
    )
