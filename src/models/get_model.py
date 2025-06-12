import torch
import collections
from collections import OrderedDict

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

def get_model(args):
    """
    Function to get the model based on the provided arguments.
    Args:
        args: Command line arguments containing model type and other configurations.
    Returns:
        model: The initialized model.
    """
    if args.model == "SegformerB0":
        from models.segformer.model import mit_b0
        model = mit_b0()
        # Load pretrained weights if specified
        if args.load_pretrained:
            checkpoint_path = os.path.join(root_dir, 'ckpt', "mit_b0.pth")
            if os.path.exists(checkpoint_path):
                save_model = torch.load(checkpoint_path, map_location=torch.device('cuda'))
                model_dict =  model.state_dict()
                state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
                model_dict.update(state_dict)
                model.load_state_dict(model_dict, strict=False)
            else:
                print(f"Checkpoint {checkpoint_path} does not exist. Skipping loading pretrained weights.")

    elif args.model == "SegformerB0_tanh":
        from models.segformer.model_tanh import mit_b0
        model = mit_b0()

    elif args.model == "SegformerB2":
        from models.segformer.model import mit_b2
        model = mit_b2()
    else:
        raise ValueError(f"Model type {args.model} not recognized. Please choose 'SegformerB0' or 'SegformerB2'.")

    return model

def get_teacher_model():
    """
    Function to get the model based on the provided arguments.
    Args:
        args: Command line arguments containing model type and other configurations.
    Returns:
        model: The initialized model.
    """
    from models.segformer.model import mit_b2
    model = mit_b2()
    checkpoint_path = "./ckpt/segformerb2_teacher_cityscapes.pth"

    model = load_segformer_weights(model, checkpoint_path)

    return model

def load_segformer_weights(model, checkpoint_path, device='cuda'):
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cuda'))
     
    # 'module.' removal for DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    try:
        # Directly load the 'linear_fuse' parameters
        norm_keys = ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']
        for key in norm_keys:
            full_key = f'linear_fuse.norm.{key}'
            if full_key in new_state_dict:
                if key == 'weight' or key == 'bias':
                    model.linear_fuse.norm.__getattr__(key).data = new_state_dict[full_key]
                else:
                    model.linear_fuse.norm.__setattr__(key, new_state_dict[full_key])
                new_state_dict.pop(full_key)
    
    except AttributeError as e:
        print(f"Warning: Could not load norm parameters: {e}")
    
    # Remaining parameters load
    own_state = model.state_dict()
    for name, param in new_state_dict.items():
        if name not in own_state:
            print(f"Parameter {name} not found in model")
            continue
        try:
            own_state[name].copy_(param)
        except Exception as e:
            print(f"Error loading parameter {name}: {e}")
    
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load Segformer model weights.")
    parser.add_argument('--model_type', type=str, required=True, help='Type of Segformer model to load (e.g., Segformer-b0, Segformer-b2)')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--device', type=str, default='cpu', help='Device to load the model on (e.g., cpu, cuda)')
    
    args = parser.parse_args()
    
    model = get_model(args)
    model = load_segformer_weights(model, args.checkpoint_path, args.device)
    breakpoint()