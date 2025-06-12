import sys
import os
import argparse

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from torch.utils.data import DataLoader
from src.data.dataset import Traindataset, Testdataset, Cotransform, get_test_transform

import cv2

from PIL import Image, ImageOps
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage, InterpolationMode
from src.data.transform import Relabel, ToLabel, Colorize

# args.data_dir = "../data/cityscapes"

def get_dataset(args):
    """
    Function to get the dataset based on the provided arguments.
    
    Args:
        args: Command line arguments containing data directory and other configurations.
    
    Returns:
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        data_collator: The data collator for batching.
    """

    train_cotransformer = Cotransform(augment=args.augmentation, height=args.input_size, model=args.model)
    train_dataset = Traindataset(args.datadir, co_transform=train_cotransformer, subset='train')

    val_cotransformer = Cotransform(augment=False, height=args.input_size, model=args.model)
    val_dataset = Traindataset(args.datadir, co_transform=val_cotransformer, subset='val')

    return train_dataset, val_dataset

# 예시를 위해 넣어둠(main에서 사용)
def get_arguments():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description="Cityscapes Dataloader Arguments")
    
    # Dataset
    parser.add_argument("--data-dir", type=str, default="../data/cityscapes",
                        help="Path to the directory containing the Cityscapes dataset")
    parser.add_argument("--size", type=int, default=512,
                        help="Comma-separated string with height and width of images")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of images sent to the network in one step")
    parser.add_argument("--augmentation", type=bool, default=True,
                        help="Apply data augmentation")
    parser.add_argument("--model", type=str, default="SegformerB0",
                        help="Model type to use for training")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    train_loader = get_train_dataloader(args)
    val_loader = get_val_dataloader(args)
    test_loader = get_test_dataloader(args)

    imgs, labels = next(iter(test_loader))
    breakpoint()