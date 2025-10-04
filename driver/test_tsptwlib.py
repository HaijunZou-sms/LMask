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
from lmask.envs.tsptw.local_search import local_search_from_data
import lmask.models.policy
from lmask.utils.data_utils import read_tsptw_instance, read_dumas_distance_matrix
from lmask.utils.metric_utils import  get_filtered_max
from lmask.utils.utils import seed_everything

def convert_length(x):
    return x if math.isinf(x) else round(x)


logging.getLogger("rl4co").setLevel(logging.ERROR)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="Attribute.*is an instance of `nn.Module`")

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--env_name", type=str, default="tsptw-lazymask")
    parser.add_argument("--policy_name", type=str, default="TSPTWRIEPolicy", help="class name of the policy")
    parser.add_argument("--checkpoint", type=str, default="./pretrained/tsptw/tsptw50-medium.pth")
    parser.add_argument("--test_dir", type=str, default="./data/dumas")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--max_backtrack_steps", type=int, default=5000)
    parser.add_argument("--local_search", action="store_true", help="whether to use local search")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_trials", type=int, default=3)
    args = parser.parse_args()

    csv_file = f"{args.save_dir}/tsptwlib-tsptw50.csv"
    print(f"Test data from {args.test_dir}")
    print(f"Load model from {args.checkpoint}")
    print(f"Use environment {args.env_name}")
    print(f"Max backtrack steps: {args.max_backtrack_steps}")
    print(f"Save results to {csv_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    env = get_env(args.env_name, max_backtrack_steps=args.max_backtrack_steps, lookahead_step=2, phase="validation", get_reward_by_distance=True)
    policy = getattr(lmask.models.policy, args.policy_name)()
    policy.load_state_dict(torch.load(args.checkpoint))
    policy.to(device).eval()

    os.makedirs(args.save_dir, exist_ok=True)
    test_dir = args.test_dir
    locations_dir = os.path.join(test_dir, "locations")
    distances_dir = os.path.join(test_dir, "distances")
    location_files = [f for f in natsorted(os.listdir(locations_dir)) if f.endswith(".txt")]
    instance_names = [os.path.splitext(f)[0] for f in location_files]
    
    scaler = 50.0
    for instance_name in instance_names:
        location_path = os.path.join(locations_dir, f"{instance_name}.txt")
        distance_path = os.path.join(distances_dir, f"{instance_name}.txt")
        td = read_tsptw_instance(location_path)
        td["distance_matrix"] = read_dumas_distance_matrix(distance_path)
        td["duration_matrix"] = td["distance_matrix"] + td["service_time"].unsqueeze(-1)
        td = td / scaler

        start = time.time()
        out, td = env.rollout(td, policy, return_td = True)
        # local_search
        if args.local_search:
            actions = td["node_stack"][:, 1:] 
            actions, cost, sol_feas = local_search_from_data(td.cpu(), actions.cpu().numpy(), num_workers = args.num_workers, max_trials = args.max_trials, return_only_solutions=False)
            reward, sol_feas = - cost.unsqueeze(0), sol_feas.unsqueeze(0)

        else:
            reward, sol_feas = out["reward"], out["sol_feas"]
        inference_time = time.time() - start
        max_aug_reward = get_filtered_max(reward, sol_feas)
        ins_feas = sol_feas.any(dim=1).float().item()
        
        print(f"Instance {instance_name} finished, best cost: {- max_aug_reward.item()*scaler: .0f}")
        df = pd.DataFrame(
            {
                "Problem": [instance_name],
                "Length": [convert_length(-max_aug_reward.item() * scaler)],
                "Ins_feas": [ins_feas],
                "Sovling Time": [round(inference_time, 2)],
            }
        )
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df.to_csv(csv_file, mode="a", header=not os.path.exists(csv_file), index=False)
