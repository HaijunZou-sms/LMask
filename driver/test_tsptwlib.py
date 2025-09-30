import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
import time
import math
import argparse
import warnings
import logging

import torch
import pandas as pd
import numpy as np
from natsort import natsorted


from lmask.envs import get_env
import lmask.models.policy
from lmask.utils.data_utils import read_tsptw_instance
from lmask.utils.metric_utils import  get_filtered_max
from lmask.utils.utils import seed_everything

def convert_length(x):
    return x if math.isinf(x) else int(x)


logging.getLogger("rl4co").setLevel(logging.ERROR)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="Attribute.*is an instance of `nn.Module`")

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--env_name", type=str, default="tsptw-lazymask")
    parser.add_argument("--policy_name", type=str, default="TSPTWPolicy", help="class name of the policy")
    parser.add_argument("--checkpoint", type=str, default="./pretrained/tsptw/tsptw50-hard.pth")
    parser.add_argument("--test_dir", type=str, default="./data/dumas")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--max_backtrack_steps", type=int, default=1000)
    parser.add_argument("--local_search", action="store_true", help="whether to use local search")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_trials", type=int, default=100)
    args = parser.parse_args()

    csv_file = f"{args.save_dir}/tsptwlib_{args.env_name}.csv"
    print(f"Test data from {args.test_dir}")
    print(f"Load model from {args.checkpoint}")
    print(f"Use environment {args.env_name}")
    print(f"Max backtrack steps: {args.max_backtrack_steps}")
    print(f"Save results to {csv_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    # env = get_env(args.env_name, max_backtrack_steps=args.max_backtrack_steps, phase="validation")
    env = get_env(args.env_name, max_backtrack_steps=args.max_backtrack_steps, lookahead_step=1)
    policy = getattr(lmask.models.policy, args.policy_name)()
    policy.load_state_dict(torch.load(args.checkpoint))
    policy.to(device).eval()

    os.makedirs(args.save_dir, exist_ok=True)
    test_dir = args.test_dir
    file_list = [os.path.join(test_dir, f) for f in natsorted(os.listdir(test_dir)) if f.endswith(".txt")]
    file_list = ["./data/dumas/n20w20.003.txt"]
    for file in file_list:
        instance_name = os.path.splitext(os.path.basename(file))[0]
        print(f"Test {instance_name}")

        td = read_tsptw_instance(file)
        loc_scaler = 1.0
        td.update(
            {
                "locs": td["locs"] / loc_scaler,
                "service_time": td["service_time"] / loc_scaler,
                "time_windows": td["time_windows"] / loc_scaler,
            }
        )
        
        # d_i0 = (td["locs"][:, 1:] - td["locs"][:, 0:1]).norm(dim=-1)
        # td["time_windows"][:, 0, 1] = (d_i0 + td["time_windows"][:, 1:, 1]).max(dim=-1).values
        
        start = time.time()
        out = env.rollout(
            td,
            policy,
            local_search=args.local_search,
            num_workers=args.num_workers,
            max_trials=args.max_trials,
        )
        inference_time = time.time() - start

        sol_feas, reward = out["sol_feas"], out["reward"]
        ins_feas = sol_feas.any(dim=tuple(range(1, sol_feas.dim()))).float().item()
        max_aug_reward = get_filtered_max(reward, sol_feas)

        df = pd.DataFrame(
            {
                "Problem": [instance_name],
                "Length": [convert_length(-max_aug_reward.item() * loc_scaler)],
                "Ins_feas": [ins_feas],
                "Sovling Time": [round(inference_time, 2)],
            }
        )
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df.to_csv(csv_file, mode="a", header=not os.path.exists(csv_file), index=False)
