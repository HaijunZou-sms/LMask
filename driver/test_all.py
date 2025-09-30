import os
import sys
import argparse
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
import csv
from driver.test import test_model_on_random_dataset
from lmask.utils.utils import infer_default_cofigs


def run_tests(data_dir="./data/random", checkpoint_dir="./pretrained"):
    # Create a structure to define all test configurations
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
    csv_path = f"./results/main_results.csv"

    print(f"Running tests for {len(test_configs)} configurations...")
    print(f"Results will be saved to {csv_path}")

    results = []

    for i, config in enumerate(test_configs):
        print(f"\n[{i+1}/{len(test_configs)}] Testing {config['problem_type']}{config['problem_size']} {config['hardness']}...")

        # Get paths and settings using utility function
        inferred_configs = infer_default_cofigs(
            problem=config["problem_type"],
            problem_size=config["problem_size"],
            hardness=config["hardness"],
            seed=2025,
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
        )

        # If any of the parameters are not provided in the config, use the inferred values
        env_name = config.get("env_name", inferred_configs["env_name"])
        policy_name = config.get("policy_name", inferred_configs["policy_name"])
        checkpoint = config.get("checkpoint", inferred_configs["checkpoint"])
        test_path = config.get("test_path", inferred_configs["test_path"])

        # Run test and collect metrics
        try:
            metrics = test_model_on_random_dataset(
                env_name=env_name,
                policy_name=policy_name,
                test_path=test_path,
                checkpoint=checkpoint,
                max_backtrack_steps=config["max_backtrack_steps"],
                lookahead_step=2,
                verbose=True,
                batch_size=2500,
                seed=2025,
            )

            # Add configuration details to metrics and convert to required format
            sol_infeas_rate = 1 - metrics["sol_feas_rate"]
            ins_infeas_rate = 1 - metrics["ins_feas_rate"]
            obj_value = -metrics["avg_reward"]  # Negative of reward as requested
            gap_percent = metrics["avg_gap"]

            result = {
                "problem_type": config["problem_type"],
                "problem_size": config["problem_size"],
                "hardness": config["hardness"],
                "sol_infeas_rate": f"{sol_infeas_rate:.2%}",
                "ins_infeas_rate": f"{ins_infeas_rate:.2%}",
                "Obj": f"{obj_value:.2f}",
                "Gap": f"{gap_percent:.2%}",
                "Time": f"{int(metrics['inference_time'])}",
            }

            results.append(result)
            print(f"Completed test for {config['problem_type']}{config['problem_size']} {config['hardness']}")
        except Exception as e:
            print(f"Error running test for {config['problem_type']}{config['problem_size']} {config['hardness']}: {str(e)}")

    if results:
        # Make sure columns are in the specified order
        fieldnames = ["problem_type", "problem_size", "hardness", "sol_infeas_rate", "ins_infeas_rate", "Obj", "Gap", "Time"]

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nAll tests completed. Results saved to {csv_path}")
    else:
        print("No results were generated.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
    )
    warnings.filterwarnings("ignore", message="Attribute.*is an instance of `nn.Module`")
    warnings.filterwarnings("ignore", message="Unused keyword arguments:.*")
    parser = argparse.ArgumentParser(description="Run tests on all datasets")
    parser.add_argument("--data_dir", type=str, default="./data/random", help="Directory containing the test data")
    parser.add_argument("--checkpoint_dir", type=str, default="./pretrained", help="Directory containing the pretrained models")
    args = parser.parse_args()

    run_tests(data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir)
