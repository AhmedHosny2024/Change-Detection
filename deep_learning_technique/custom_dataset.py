import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms 
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import torch
from pylab import *
from deep_learning_technique.config import *
# train_transform = [            
#     transforms.RandomRotation((360,360), expand=False, center=None),
#     transforms.RandomVerticalFlip(p=1),
#     transforms.RandomHorizontalFlip(p=1),
#     transforms.RandomRotation((180,180), expand=False, center=None),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#     transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
#     transforms.Resize((ISIZE,ISIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ]
train_transform = [            
    transforms.RandomRotation((360,360), expand=False, center=None),
    transforms.RandomVerticalFlip(p=1),
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomRotation((180,180), expand=False, center=None),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    # transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.Resize((ISIZE,ISIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
7]
test_transform = [            
    transforms.Resize((ISIZE,ISIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5 )),
    # transforms.Normalize((7.78778377, 0.7776678, 70.51143379), (0.08119602 ,0.08063477, 0.08069688 ))78
    ]
class Custom_dataset(Dataset):
    def __init__(self, folder_Path,transform_type="train"):
        super(Custom_dataset, self).__init__()
        self.folder_A = folder_Path+"/A"
        self.folder_B = folder_Path+"/B"
        self.labels_folder = folder_Path+"/label"
        self.files_A = sorted(os.listdir(self.folder_A))
        self.files_B = sorted(os.listdir(self.folder_B))        
        self.labels_files = sorted(os.listdir(self.labels_folder))
        self.transform_type = transform_type
        self.file_size = len(self.files_A)

    def __getitem__(self, idx):

        x1 = Image.open(os.path.join(self.folder_A, self.files_A[idx]))
        x2 = Image.open(os.path.join(self.folder_B, self.files_B[idx]))
        gt = Image.open(os.path.join(self.labels_folder, self.labels_files[idx]))
        if(self.transform_type=="train"):
            k = np.random.randint(4)        
            x1 = train_transform[k](x1);x2 = train_transform[k](x2);gt = train_transform[k](gt);
            x1 = train_transform[4](x1);x2 = train_transform[4](x2);gt = train_transform[4](gt);
            x1 = train_transform[5](x1);x2 = train_transform[5](x2);gt = train_transform[5](gt);
            x1 = train_transform[6](x1);x2 = train_transform[6](x2);
        else:
            x1 = test_transform[0](x1);x2 = test_transform[0](x2);gt = test_transform[0](gt);
            x1 = test_transform[1](x1);x2 = test_transform[1](x2);gt = test_transform[1](gt);
            x1 = test_transform[2](x1);x2 = test_transform[2](x2);
        return x1, x2, gt

    def __len__(self):
        return self.file_size
    
# test the custom dataset
if __name__ == "__main__":
    # create a dataset object
    dataset = Custom_dataset(TRAIN_FOLDER_PATH)
    # create a dataloader object
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # get the first batch
    for x1, x2, gt in dataloader:
        print(x1.shape, x2.shape, gt.shape)
        break
    # display the first image
    cv2.imshow('x1', x1[0].permute(1, 2, 0).numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('x2', x2[0].permute(1, 2, 0).numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('gt', gt[0].permute(1, 2, 0).numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()