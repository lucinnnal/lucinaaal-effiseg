import os
import importlib
import time
from PIL import Image
from configs.eval_arguements import get_arguments

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor, ToPILImage

from src.data.loader import get_val_dataloader
from src.data.transform import Relabel, ToLabel, Colorize
from src.utils.iouEval import iouEval, getColorEntry, getColorEntry

from src.models.segformer.model import mit_b0, mit_b2
from src.models.get_model import get_model, load_segformer_weights

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"device : {device}")
    set_seed(42)

    if(not os.path.exists(args.datadir)): 
        print ("Error: datadir could not be loaded")
        
    # Dataloder
    loader = get_val_dataloader(args)

    # Load model to evaluate, if multiple GPUs are available, use DataParallel
    model = get_model(args)
    model = load_segformer_weights(model, args.weightspath, device=device)

    print ("Model and weights LOADED successfully")
    model.to(device)
    model.eval()

    iouEvalVal = iouEval(args.num_classes) # Metric class load

    start = time.time()

    for step, batch in enumerate(loader):
        images = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device).unsqueeze(1)

        with torch.no_grad():
            outputs = model(images)

        preds = torch.argmax(outputs['logits'], dim=1, keepdim=True)
        iouEvalVal.addBatch(preds, labels)

        # print (step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    args = get_arguments()
    main(args)