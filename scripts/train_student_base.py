import os
import sys
import warnings
warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import random
import numpy as np
import torch
from transformers import TrainingArguments
import wandb

from configs.train_arguements import get_arguments

from src.data.get_dataset import get_dataset
from src.models.get_model import get_model
from src.trainer import BaseTrainer
from src.utils.compute_metrics import compute_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"device : {device}")
    set_seed(42)

    ## =================== Data =================== ##
    train_dataset, val_dataset = get_dataset(args)

    ## =================== Base_Model =================== ##
    model = get_model(args)
    model.to(device)

    ## =================== Trainer =================== ##
    wandb.init(project='Effiseg', name=f'baseline')

    training_args = TrainingArguments(
        output_dir=f"./ckpt/{args.save_dir}",
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        eval_accumulation_steps = 100,
        per_device_train_batch_size=args.per_device_train_batch_size, 
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate, # default optimizer : AdamW with betas=(0.9, 0.999), eps=1e-8
        lr_scheduler_type="polynomial",
        logging_steps=100,
        metric_for_best_model="mean_iou",
        save_strategy="steps",
        save_total_limit=None,
        save_steps=args.save_steps,
        remove_unused_columns=False,
        report_to="wandb",
        dataloader_num_workers=0,
    )

    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=None,
    )

    import time
    
    start_time = time.time()

    trainer.train()

    end_time = time.time()
    total_time = end_time - start_time

    wandb.log({"total_training_time_sec": total_time, "total_training_time_min": total_time / 60})


if __name__=="__main__":

    args = get_arguments()
    main(args)