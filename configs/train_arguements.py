import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))  # 한 단계 상위 디렉토리
sys.path.append(parent_dir)

def get_arguments():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description="Training Arguments")
    
    # Dataset
    parser.add_argument("--datadir", type=str, default="/home/urp_jwl/.vscode-server/data/Effiseg/src/data/cityscapes", help="Path to the directory containing the dataset")
    parser.add_argument("--save_dir", type=str, default="SegformerB0_cityscapes", help="Directory to save the model checkpoints")
    parser.add_argument("--eval_strategy", type=str, default="epoch", help="Evaluation strategy to use during training")
    parser.add_argument("--model", type=str, default="SegformerB0", help="Model type to use for training")
    parser.add_argument("--student_stage_channels", type=list, default=[32, 64, 160, 256])
    parser.add_argument("--teacher_stage_channels", type=list, default=[64, 128, 320, 512])
    parser.add_argument("--embeds", type=int, default=5, help="How many stages will be included for the patch embedding loss")
    parser.add_argument("--load_pretrained", type=bool, default=False, help="Load pretrained weights for the model")
    parser.add_argument("--augmentation", type=bool, default=True, help="Apply data augmentation during training")
    parser.add_argument("--input_size", type=int, default=512, help="Length of the shorter side of the image")
    parser.add_argument("--eval_steps", type=int, default=200, help="Number of steps between evaluations")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio for learning rate scheduling")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size per device for evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=200, help="Number of epochs to train the model")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=100000, help="Number of steps between model saves")

    
    return parser.parse_args()