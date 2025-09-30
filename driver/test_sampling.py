import time
import os
import sys
import argparse
import warnings
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

from driver.test import test_model_on_random_dataset
from lmask.utils.utils import infer_default_cofigs


def run_sampling_tests(seeds=None, data_dir="./data/random", checkpoint_dir="./pretrained", sample_size = 10):
    seeds = [2025]

    # Create a structure to define test configurations (only size 50)
    test_configs = [
        # TSPTW size 50
        {
            "problem_type": "tsptw",
            "problem_size": 50,
            "hardness": "easy",
            "max_backtrack_steps": 100,
        },
        {
            "problem_type": "tsptw",
            "problem_size": 50,
            "hardness": "medium",
            "max_backtrack_steps": 100,
            "env_name": "tsptw-train",
            "policy_name": "TSPTWRIEPolicy",
        },
        {
            "problem_type": "tsptw",
            "problem_size": 50,
            "hardness": "hard",
            "max_backtrack_steps": 200,
        },
        # TSPTW size 100
        {
            "problem_type": "tsptw",
            "problem_size": 100,
            "hardness": "easy",
            "max_backtrack_steps": 150,
        },
        {
            "problem_type": "tsptw",
            "problem_size": 100,
            "hardness": "medium",
            "max_backtrack_steps": 200,
        },
        {
            "problem_type": "tsptw",
            "problem_size": 100,
            "hardness": "hard",
            "max_backtrack_steps": 300,
        },
        # TSPDL size 50
        {
            "problem_type": "tspdl",
            "problem_size": 50,
            "hardness": "medium",
            "max_backtrack_steps": 150,
        },
        {
            "problem_type": "tspdl",
            "problem_size": 50,
            "hardness": "hard",
            "max_backtrack_steps": 150,
        },
        # TSPDL size 100
        {
            "problem_type": "tspdl",
            "problem_size": 100,
            "hardness": "medium",
            "max_backtrack_steps": 150,
        },
        {
            "problem_type": "tspdl",
            "problem_size": 100,
            "hardness": "hard",
            "max_backtrack_steps": 150,
        },
    ]

    # Create CSV file to store results
    os.makedirs("./results", exist_ok=True)
    csv_path = f"./results/lmask_sample{sample_size}_results.csv"

    print(f"Running sampling tests with seeds {seeds} across {len(test_configs)} configurations...")
    print(f"Results will be saved to {csv_path}")

    results = []

    # Organize testing by configuration first, then by seeds
    for i, config in enumerate(test_configs):
        print(
            f"\n[{i+1}/{len(test_configs)}] Testing {config['problem_type']}{config['problem_size']} {config['hardness']}..."
        )

        # Get paths and settings using utility function
        for seed in seeds:
            print(f"  Running with seed {seed}")

            inferred_configs = infer_default_cofigs(
                problem=config["problem_type"],
                problem_size=config["problem_size"],
                hardness=config["hardness"],
                seed=seed,
                data_dir=data_dir,
                checkpoint_dir=checkpoint_dir,
            )

            env_name = config.get("env_name", inferred_configs["env_name"])
            policy_name = config.get("policy_name", inferred_configs["policy_name"])
            checkpoint = config.get("checkpoint", inferred_configs["checkpoint"])
            test_path = config.get("test_path", inferred_configs["test_path"])
            
            if config["problem_size"] == 100 and sample_size >=30:
                batch_size = 1200
            else:
                batch_size = 2500
            # Run test with sampling decoding
            try:
                metrics = test_model_on_random_dataset(
                    env_name=env_name,
                    policy_name=policy_name,
                    test_path=test_path,
                    checkpoint=checkpoint,
                    max_backtrack_steps=config["max_backtrack_steps"],
                    look_ahead_step=2,
                    verbose=True,
                    batch_size=batch_size,
                    seed=seed,
                    decode_type='sampling',
                    num_samples=sample_size,
                )

                # Add configuration details to metrics and convert to required format
                sol_infeas_rate = 1 - metrics["sol_feas_rate"]
                ins_infeas_rate = 1 - metrics["ins_feas_rate"]
                obj_value = -metrics["avg_reward"]  # Negative of reward
                gap_percent = metrics["avg_gap"]

                result = {
                    "problem_type": config["problem_type"],
                    "problem_size": config["problem_size"],
                    "hardness": config["hardness"],
                    "seed": seed,
                    "sol_infeas_rate": f"{sol_infeas_rate:.3%}",
                    "ins_infeas_rate": f"{ins_infeas_rate:.3%}",
                    "Obj": f"{obj_value:.3f}",
                    "Gap": f"{gap_percent:.3%}",
                    "Time": f"{int(metrics['inference_time'])}",
                }

                results.append(result)
                print(f"  Completed test for seed {seed}")
            except Exception as e:
                print(f"  Error running test for seed {seed}: {str(e)}")

    if results:
        # Define columns order
        fieldnames = [
            "problem_type",
            "problem_size",
            "hardness",
            "seed",
            "sol_infeas_rate",
            "ins_infeas_rate",
            "Obj",
            "Gap",
            "Time",
        ]

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nAll tests completed. Results saved to {csv_path}")
    else:
        print("No results were generated.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="Attribute.*is an instance of `nn.Module`")
    warnings.filterwarnings("ignore", message="Unused keyword arguments:.*")
    parser = argparse.ArgumentParser(description="Run sampling tests on size 50 datasets")
    parser.add_argument("--data_dir", type=str, default="./data/random", help="Directory containing the test data")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./pretrained", help="Directory containing the pretrained models"
    )
    parser.add_argument("--sample_size", type=int, default=10, help="sample size")
    args = parser.parse_args()

    run_sampling_tests(data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir, sample_size=args.sample_size)
