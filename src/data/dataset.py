import random
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import os
import os.path as osp
import sys

import torch
from torch.utils.data import Dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))  # 한 단계 상위 디렉토리
sys.path.append(parent_dir)
import cv2
from PIL import Image, ImageOps
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage, InterpolationMode
from src.data.transform import Relabel, ToLabel, Colorize

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(root_dir)

from models.segformer.model import mit_b0, mit_b2, load_model_weights

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class Traindataset(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.labels_root += subset

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()


        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return {
            "pixel_values" : image, 
            "labels": label.squeeze(0)
        }

    def __len__(self):
        return len(self.filenames)

class Cotransform(object):
    def __init__(self, augment=True, height=512, model='SegformerB0'): # Evaluation시에 False로 설정
        self.augment = augment
        self.height = height
        self.model = model
        pass
    def __call__(self, input, target):
        input =  Resize(self.height, Image.BILINEAR)(input)
        # Input image is resized to 512x1024
        W,H = input.size
        if self.model.startswith('Segformer'):
            target = Resize((int(H/4+0.5),int(W/4+0.5)), Image.NEAREST)(target) # Target label is resized to 128x256 (1/4,1/4)
        else:
            assert 'model not supported'
        if(self.augment):
            # Flip horizontally for 50% if augmentation is True
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Shift the image and label randomly
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255)
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target

# Test
class Testdataset(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, subset='test'):
        self.images_root = os.path.join(root, 'leftImg8bit/' + subset)
        self.labels_root = os.path.join(root, 'gtFine/' + subset)

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)

def get_test_transform(inpuut_size=512, target_size=128):
    input_transform = Compose([Resize(inpuut_size, interpolation=InterpolationMode.BILINEAR), ToTensor()])
    target_transform = Compose([Resize(target_size, interpolation=InterpolationMode.NEAREST), ToLabel(), Relabel(255, 19)])
    
    return input_transform, target_transform

if __name__ == '__main__':
    set_seed(42)

    """
    # for training and evaluation
    transform = Cotransform(augment=False, height=512, model='SegformerB0')
    dataset = Traindataset(root='../data/cityscapes', co_transform=transform, subset='train')
    image, label = dataset[0]
    breakpoint()
    """

    # for testing
    input_transform, target_transform = get_test_transform()
    dataset = Testdataset(root='../data/cityscapes', input_transform=input_transform, target_transform=target_transform, subset='test')
    image, label = dataset[0]
