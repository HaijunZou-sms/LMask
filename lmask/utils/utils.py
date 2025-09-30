import time
import torch
import random
import numpy as np


def infer_default_cofigs(problem, problem_size, hardness, seed=2025, data_dir="./data/random", checkpoint_dir="./pretrained"):
    """
    Infer parameters from problem information

    Args:
        problem: Problem type ('tspdl' or 'tsptw')
        problem_size: Problem size (50 or 100)
        hardness: Problem difficulty ('easy', 'medium', or 'hard')
        seed: Random seed for test path
        data_dir: Directory containing the test data (default: ./data/random)
        checkpoint_dir: Directory containing the pretrained models (default: ./pretrained)
    Returns:
        dict: Dictionary with inferred parameters
    """
    policy_name = f"{problem.upper()}Policy"
    checkpoint = f"{checkpoint_dir}/{problem}/{problem}{problem_size}-{hardness}.pth"
    test_path = f"{data_dir}/{problem}/test/{problem}{problem_size}_test_{hardness}_seed2025.npz"
    env_name = f"{problem}-lazymask"

    return {"policy_name": policy_name, "checkpoint": checkpoint, "test_path": test_path, "env_name": env_name}


def seed_everything(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class TimeEstimator:
    def __init__(self):
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count - 1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total - count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time * 60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time * 60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        print("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(count, total, elapsed_time_str, remain_time_str))


def num_param(model):
    nb_param = 0
    for param in model.parameters():
        nb_param += param.numel()
    print("Number of Parameters: {}".format(nb_param))
