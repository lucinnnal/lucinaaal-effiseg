from ptflops import get_model_complexity_info

import warnings
warnings.filterwarnings("ignore")

import os
import sys

import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

from configs.train_arguements import get_arguments
from src.data.get_dataset import get_dataset
from src.models.get_model import get_model

import torch
import time

def profile_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(args)
    model.to(device)
    model.eval()

    input_shape = (3, args.input_size, args.input_size * 2)  

    # FLOPs (total calculation amount per inference) + Param # (parameters # for training)
    flops, params = get_model_complexity_info(
        model, input_shape, as_strings=False, print_per_layer_stat=False
    )

    print(f"FLOPs : {flops}")
    print(f"Params: {params}")

    # FPS -> How many images can model process per second?
    # Dummy variable
    input_tensor = torch.randn(1, 3, args.input_size, args.input_size * 2).cuda()


    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor, is_feat=False)

    num_runs = 100
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor, is_feat=False)
    torch.cuda.synchronize()
    end = time.time()

    fps = num_runs / (end - start)
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    args = get_arguments()
    profile_model(args)