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

train_transform = [            
    transforms.RandomRotation((360,360), expand=False, center=None),
    transforms.RandomVerticalFlip(p=1),
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomRotation((180,180), expand=False, center=None),
    transforms.RandomRotation((90, 90), expand=False, center=None),
    transforms.RandomVerticalFlip(p=1),
    transforms.RandomHorizontalFlip(p=1),
    transforms.Resize((ISIZE,ISIZE)),
    # transforms.RandomResizedCrop((ISIZE, ISIZE), scale=(0.8, 1.0)),  # Adding zooming
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
test_transform = [            
    transforms.Resize((ISIZE,ISIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5 )),
    ]
class Custom_dataset(Dataset):
    def __init__ (self, folder_Path,transform_type="train"):
        super(Custom_dataset, self).__init__()
        self.folder_A = folder_Path+"/A"
        self.folder_B = folder_Path+"/B"
        self.labels_folder = folder_Path+"/label"
        self.files_A = sorted(os.listdir(self.folder_A))
        self.files_B = sorted(os.listdir(self.folder_B))        
        self.labels_files = sorted(os.listdir(self.labels_folder))
        self.transform_type = transform_type
        self.file_size = len(self.files_A)
        if '.ipynb_checkpoints' in self.files_A:
            self.files_A.remove('.ipynb_checkpoints')
        if '.ipynb_checkpoints' in self.files_B:
            self.files_B.remove('.ipynb_checkpoints')
        if '.ipynb_checkpoints' in self.labels_files:
            self.labels_files.remove('.ipynb_checkpoints')

    def __getitem__(self, idx):        
        x1 = Image.open(os.path.join(self.folder_A, self.files_A[idx]))
        x2 = Image.open(os.path.join(self.folder_B, self.files_B[idx]))
        gt = Image.open(os.path.join(self.labels_folder, self.labels_files[idx]))
        if(self.transform_type=="train"):
            k = np.random.randint(7)        
            x1 = train_transform[k](x1);x2 = train_transform[k](x2);gt = train_transform[k](gt);
            x1 = train_transform[7](x1);x2 = train_transform[7](x2);gt = train_transform[7](gt);
            x1 = train_transform[8](x1);x2 = train_transform[8](x2);gt = train_transform[8](gt);
            x1 = train_transform[9](x1);x2 = train_transform[9](x2);
        else:
            x1 = test_transform[0](x1);x2 = test_transform[0](x2);gt = test_transform[0](gt);
            x1 = test_transform[1](x1);x2 = test_transform[1](x2);gt = test_transform[1](gt);
            x1 = test_transform[2](x1);x2 = test_transform[2](x2);
        return x1, x2, gt

    def __len__(self):
        return self.file_size
    
    def get_files_name(self):
        return self.labels_files
    
# test the custom dataset
if __name__ == "__main__":
    # create a dataset object
    dataset = Custom_dataset(TRAIN_FOLDER_PATH)
    # create a dataloader object
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # get the first batch
    i=0
    for x1, x2, gt in dataloader:
        print(x1.shape, x2.shape, gt.shape)
        if i == 50:
            break
        i+=1
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