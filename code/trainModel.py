import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import get_dataset
from Unet import unet
from segmentation_models_pytorch.losses import JaccardLoss
import segmentation_models_pytorch as smp
#from smp.utils.losses import DiceLoss
#from smp.utils.metrics import IoU
import matplotlib.pyplot as plt
from trainTestSplit import trainTestSplit
import albumentations as A
from albumentations.pytorch import ToTensorV2

'''Train model'''

#seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#assign train/val folder
path = '../dataset/public'
subjects = ['S1', 'S2', "S3", "S4"]
test_subject = ['S5']

#train-test split
train_images, val_images, train_label, val_label, test_images = trainTestSplit(path, subjects, test_subject)


#hyper parameters
BATCH_SIZE = 8
LR = 0.001
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#ref from github
def binary_iou(input, target):
    input_area = input.sum()
    target_area = target.sum()
    area_union  = torch.logical_or(input, target).sum()
    if area_union  == 0:
        return torch.ones(1)
    area_inter = input_area + target_area - area_union
    iou = area_inter / area_union
    return iou

def train():
    #in __getitem__, the Dataset class will use that function to augment an image and a mask and return their augmented versions
    train_transform = A.Compose(
    [
        #A.Resize(240, 320),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        #A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
    )
    val_transform = A.Compose([
    #A.Resize(240, 320),
    A.Normalize(),
    ToTensorV2()]
    )
    train_dataset = get_dataset(train_images, train_label, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataset = get_dataset(val_images, val_label, val_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataset = get_dataset(test_images, train_label, None)
    #test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    folder = "saved_images/"

    model = unet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)


    best_valid_iou = 0.0
    for epoch in range(EPOCHS):
        #train
        train_loss = 0.0
        train_acc = 0.0
        train_iou = 0.0
        train_total = 0.0
        model.train()
        for step, (image, label) in enumerate(train_loader):
            image = image.to(DEVICE)  #torch.Size([10, 1, 480, 640])
            #print(image.shape)
            label = label.to(DEVICE).to(torch.int64)  #torch.Size([10, 480, 640])
            #print(label.shape)
            #np.savetxt("label.txt", label.reshape(3, -1))

            out = model(image)        #torch.Size([10, 2, 480, 640])
            #print(out)

            optimizer.zero_grad()
            iouLoss = JaccardLoss(mode= "multiclass", classes=[1])
            loss = iouLoss(out, label)   #tensor(0.9796, grad_fn=<MeanBackward0>)
            #print(loss)

            loss.backward()
            optimizer.step()

            _, pred = torch.max(out.data, 1)                              #out: torch.Size([8, 2, 480, 640]) /
            acc = pred.eq(label.data).cpu().sum() / np.prod(label.shape)  #tensor(0.7847)
            #print("acc", acc)
            #count iou
            iou = binary_iou(pred, label)
            train_loss += loss.item() * image.shape[0]
            train_acc += acc.item() * image.shape[0]
            train_iou += iou.item() * image.shape[0]
            train_total += image.shape[0]
            print(f"Epoch ({epoch+1}) Step ({step+1}/{len(train_loader)})  Train loss: {loss.item()}  Train acc: {acc.item()}  Train iou: {iou.item()}", end='\r')

        train_loss = train_loss / train_total
        train_acc = train_acc / train_total
        train_iou = train_iou / train_total

        #valid
        valid_loss = 0.0
        valid_acc = 0.0
        valid_iou = 0.0
        valid_total = 0.0
        model.eval()
        with torch.no_grad():
            for step, (image, label) in enumerate(valid_loader):
                image = image.to(DEVICE)
                label = label.to(DEVICE).to(torch.int64)

                out = model(image)
                iouLoss = JaccardLoss(mode= "multiclass", classes=[1])
                loss = iouLoss(out, label)
                _, pred = torch.max(out.data, 1)
                acc = pred.eq(label.data).cpu().sum() / np.prod(label.shape)
                iou = binary_iou(pred, label)
                valid_loss += loss.item() * image.shape[0]
                valid_acc += acc.item() * image.shape[0]
                valid_iou += iou.item() * image.shape[0]
                valid_total += image.shape[0]
                print(f"Epoch ({epoch+1}) Step ({step+1}/{len(valid_loader)})  Valid loss: {loss.item()}  Valid acc: {acc.item()}  Valid iou: {iou.item()}", end='\r')

        valid_loss = valid_loss / valid_total
        valid_acc = valid_acc / valid_total
        valid_iou = valid_iou / valid_total

        print(f"Epoch ({epoch+1})  Train loss: {train_loss}  Train acc: {train_acc}  Train iou: {train_iou}  Valid loss: {valid_loss}  Valid acc: {valid_acc}  Valid iou: {valid_iou}")
        if valid_iou > best_valid_iou:
            best_valid_iou = valid_iou
            print("Saving model")
            torch.save(model.state_dict(), "model_best.pth")


if __name__ == '__main__':
    train()




'''
plt.hist(out.data.numpy().ravel(), 20, [0,20])
            plt.savefig('out.jpg')
            plt.show()

'''
