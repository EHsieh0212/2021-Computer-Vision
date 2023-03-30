from collections import defaultdict
import copy
import random
import os
import shutil
from urllib.request import urlretrieve
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
#import ternausnet.models
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader



def trainTestSplit(path, subjects, test_subject, x=0.9):
    total_length = 19426  #pre_calculation
    train_size = int(total_length*x)
    images = []
    masks = []
    test_images = []
    for s in subjects:
        for sequence in range(1, 27):
            folder = os.path.join(path, s, f'{sequence:02d}')
            folder_count = len([i for i in os.listdir(folder)])//2
            if s == 'S5':
                test_images.append(os.path.join(folder, f"{idx}.jpg"))
            for idx in range(folder_count):
                images.append(os.path.join(folder, f"{idx}.jpg"))
                masks.append(os.path.join(folder, f"{idx}.png"))
    random.seed(42)
    #random.shuffle(images)
    #random.shuffle(masks)
    print(train_size)
    train_images = images[:train_size]
    train_labels = masks[:train_size]
    val_images = images[train_size:]
    val_labels = masks[train_size:]
    return train_images, val_images, train_labels, val_labels, test_images



if __name__ == "__main__":
    path = '../dataset/public'
    subjects = ['S1', 'S2', 'S3', 'S4']
    test_subject = ['S5']
    train_images, val_images, train_label, val_label, test_images = trainTestSplit(path, subjects, test_subject)
