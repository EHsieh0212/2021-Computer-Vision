import imghdr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from trainTestSplit import trainTestSplit
from Unet import unet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt


#seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#assign train/val folder
#path = '../dataset/public'
#subjects = ['S1', 'S2', "S3", "S4"]
#test_subject = ['S5']
#train-test split
#train_images, val_images, train_label, val_label, test_images = trainTestSplit(path, subjects, test_subject)

#hyper parameters
BATCH_SIZE = 8
LR = 0.001
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path="./model_best.pth"
model = unet().to(DEVICE)

if DEVICE == 'cpu':
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
else:
    model.load_state_dict(torch.load(model_path))

model.eval()

@torch.no_grad()
def my_awesome_algorithm(image):
    # print(image)

    THRESHOLD = 0.001
    #process testing input images (ref from github)
    #if len(image.shape)==2:
    #    image = np.expand_dims(image, 2)
    #elif image.shape[2]==3:
    #    image = image[..., :1]
    #print(image.shape)
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    transform = A.Compose(
    [
        #A.Resize(240, 320),
        #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        #A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        #A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
    )

    # print("start transform")
    image = transform(image=image)["image"]
    # print(image.shape)
    # print("finish transform")
    image = image[[1], ...]
    # print(image.shape)
    # print("to device")
    image = image.to(DEVICE)
    image = image.unsqueeze(0)
    # print(image.shape)
    # print("start predict")
    out = model(image)
    # print("finish predcut")

    #print(out)
    _, pred = torch.max(out.data, dim=1) #测试集有10个数据，那么训练好的网络将会预测这10个数据，得到一个10×2的矩阵
    pred = pred[0].cpu().numpy()
    #plt.hist(pred.ravel(), 20, [0,20])
    #plt.show()
    #plt.imshow(pred)
    #plt.show()
    #pred = pred[0].cpu().numpy()
    #pupil / pic area percentage
    area = pred.sum() / np.prod(pred.shape)
    #percentage > threshold: there is pupil
    conf = float(area > THRESHOLD)

    return pred, conf

