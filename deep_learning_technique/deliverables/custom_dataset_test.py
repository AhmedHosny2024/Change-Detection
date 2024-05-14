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
test_transform = [            
    transforms.Resize((ISIZE,ISIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5 )),
    ]
class Custom_dataset_test(Dataset):
    def __init__(self, folder_Path):
        super(Custom_dataset_test, self).__init__()
        self.folder_A = folder_Path+"/A"
        self.folder_B = folder_Path+"/B"
        self.labels_folder = folder_Path+"/label"
        self.files_A = sorted(os.listdir(self.folder_A))
        self.files_B = sorted(os.listdir(self.folder_B))        
        self.labels_files = sorted(os.listdir(self.labels_folder))
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
        x1 = test_transform[0](x1);x2 = test_transform[0](x2);
        x1 = test_transform[1](x1);x2 = test_transform[1](x2);
        x1 = test_transform[2](x1);x2 = test_transform[2](x2);
        return x1, x2

    def __len__(self):
        return self.file_size
    
    def get_files_name(self):
        return self.labels_files
    
# test the custom dataset
if __name__ == "__main__":
    # create a dataset object
    dataset = Custom_dataset_test(TRAIN_FOLDER_PATH)
    # create a dataloader object
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # get the first batch
    i=0
    for x1, x2 in dataloader:
        print(x1.shape, x2.shape)
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
   