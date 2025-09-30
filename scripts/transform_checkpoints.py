import torch
import os
import sys
import argparse
import warnings
import glob
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)


def transform_checkpoints(input_dir, replace_original=False):
    if replace_original:
        output_dir = input_dir
        print("Replacing original checkpoint files")
    else:
        output_dir = os.path.join(input_dir, "modified")
        print(f"Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    checkpoint_files = glob.glob(os.path.join(input_dir, "*.ckpt"))
    
    for checkpoint_file in tqdm(checkpoint_files):
        filename = os.path.basename(checkpoint_file)
        output_filename = os.path.splitext(filename)[0] + ".pth"
        output_path = os.path.join(output_dir, output_filename)
        
        checkpoint = torch.load(checkpoint_file)
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace("policy.", ""): v for k, v in state_dict.items()}
        
        torch.save(state_dict, output_path)

        # Delete original file if replacing
        if replace_original:
            os.remove(checkpoint_file)
            print(f"Replaced {checkpoint_file} with {output_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(
        description="Convert checkpoint files to .pth files with policy prefix removed"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./selected_checkpoints",
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--replace", action="store_true", help="Replace original checkpoint files with .pth files"
    )
    args = parser.parse_args()

    transform_checkpoints(args.input_dir, replace_original=args.replace)
