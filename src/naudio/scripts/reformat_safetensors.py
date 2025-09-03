from safetensors import torch as torchsft
import argparse
from pathlib import Path

t5_special_cases = {
    "shared.weight": "embed.embedding",
    "encoder.final_layer_norm.weight": "encoder.t5ln.weight"
}

def reformat_t5(input_path: str, output_path: str):
    # Load the original safetensors file
    original = torchsft.load_file(input_path)
    keys = []
    # Reformat the model weights
    reformatted = {}
    for key, value in original.items(): 
        print(f"Processing key: {key}")
        if not key.startswith("encoder.block") and key not in t5_special_cases:
            continue
        if key in t5_special_cases:
            new_key = t5_special_cases[key]
        else:
            nameofvalue = "kernel" 
            chunks = key.split('.')
            sublayer = chunks[4]
            new_key = "encoder.blocks." + chunks[2]
            if sublayer == "0": #attn layer
                new_key += ".attention"
                if chunks[5] == "layer_norm":
                    new_key += ".t5ln"
                    nameofvalue = "weight"
                else:
                    new_key += ".attention"
                    if chunks[6] == "relative_attention_bias":
                        new_key += ".relattnbias"
                        nameofvalue = "embedding"
                    else:
                        new_key += ".to" + chunks[6]
            else:
                new_key += ".ff"
                if chunks[5] == "layer_norm":
                    new_key += ".t5ln"
                    nameofvalue = "weight"
                else:
                    new_key += ".dense"
                    if chunks[6] == "wi":
                        new_key += ".wi"
                    else:
                        new_key += ".wo"
            new_key += "." + nameofvalue
        keys.append(new_key)
        reformatted[new_key] = value
    
    for k in keys:
        print(f"- {k}")
    print(keys)
    # Save the reformatted weights
    torchsft.save_file(reformatted, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Reformat a safetensors file to match the expected key structure"
    )
    parser.add_argument("input_st", help="Path to the input .safetensors file")
    parser.add_argument("output_st", help="Path to the output .safetensors file")
    parser.add_argument("model_type", help="Type of model (e.g., 't5', 'dit', 'vae')")
    args = parser.parse_args()

    if args.model_type.lower() == "t5":
        reformat_t5(args.input_st, args.output_st)
    else:
        print(f"Model type '{args.model_type}' not supported yet.")

if __name__ == "__main__":
    main()