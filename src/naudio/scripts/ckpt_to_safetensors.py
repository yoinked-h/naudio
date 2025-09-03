import argparse
import torch
from safetensors.torch import save_model

def convert_ckpt_to_safetensors(input_path: str, output_path: str):
    # Load the checkpoint on CPU
    ckpt = torch.load(input_path, map_location="cpu")
    # Save as safetensors
    save_model(ckpt, output_path)
    print(f"Converted '{input_path}' to '{output_path}'")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch .ckpt checkpoint to a .safetensors file"
    )
    parser.add_argument("input_ckpt", help="Path to the input .ckpt file")
    parser.add_argument("output_st", help="Path to the output .safetensors file")
    args = parser.parse_args()

    convert_ckpt_to_safetensors(args.input_ckpt, args.output_st)

if __name__ == "__main__":
    main()