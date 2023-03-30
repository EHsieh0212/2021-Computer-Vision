import os
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from trainTestSplit import trainTestSplit
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_dataset(images, labels, transform):
    dataset = PupilDataset(images, labels, transform)
    return dataset


class PupilDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, item):
        #one d
        image = np.asarray(Image.open(self.images[item]))
        #image = np.expand_dims(image, 2)
        #two d
        mask = np.asarray(Image.open(self.labels[item]).convert('RGB'))   #(480, 640, 3)
        mask = (mask.sum(axis=2) > 0).astype(np.int64)                   #(480, 640)   with 2 pixel values 0,1
        #print(mask.shape)

        if self.transform is not None:
            #image = np.stack((image, image, image), axis = 2)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            transformed = self.transform(image=image, mask=mask)
            #print(";;;;")
            image = transformed["image"]
            #image = np.moveaxis(image.numpy(), 0, -1)
            image = image[[1], ...]
            #image = torch.from_numpy(image.numpy())
            #print(image.shape)
            label = transformed["mask"]
            #print(label.shape)

        return image, label

    def __len__(self):
        return len(self.images)

'''
if __name__ == '__main__':
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
    DEVICE = torch.device("cpu")

    train_transform = A.Compose(
    [
        A.Resize(240, 320),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(),
        ToTensorV2(p=1),
    ]
    )
    train_dataset = get_dataset(train_images, train_label, train_transform)

    print("__")
    fimage, flabel = train_dataset.__getitem__(5)
    print("++")
    print(fimage.shape)
    print(flabel.shape)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #print(train_loader)
     for image, label in train_loader:
        print(image.shape, label)
    '''
















'''
dataset_path = '../dataset/public'
sno = ['S1', 'S2', 'S3', 'S4']
dataset = get_dataset(dataset_path, sno)
a = dataset.__len__()
print(a)
second_image, label = dataset.__getitem__(5)
print(label)
plt.imshow(label, cmap='gray')
plt.show()
plt.hist(label.ravel(), 5, [0,5])
plt.show()


from torch.utils.data import Dataset, DataLoader
dataset_path = '../dataset/public'
sno = ['S1', 'S2', 'S3', 'S4']
dataset = get_dataset(dataset_path, sno)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
for image, label in dataloader:
    print(image.shape, label.shape)
    print(image.max(), image.min())
    break


second_image, label = dataset.__getitem__(5)
#print(second_image.shape)
#print(label.shape)
print(second_image.shape)
print(label.shape)
#plt.hist(tmp.ravel(), 255, [0,255])
#plt.show()

#############################################################################
try1 = np.asarray(Image.open("158.jpg"))
try2 = np.asarray(Image.open("158.png"))
#print(try1.shape)
#print(try2.shape)
#plt.hist(try1.ravel(), 255, [0,255])
#plt.show()
'''























