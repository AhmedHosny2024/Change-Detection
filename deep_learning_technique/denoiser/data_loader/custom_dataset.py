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
from deep_learning_technique.denoiser.config import *
from deep_learning_technique.config import *
import matplotlib.pylab as plb


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
        if '.ipynb_checkpoints' in self.files_A:
            self.files_A.remove('.ipynb_checkpoints')
        if '.ipynb_checkpoints' in self.files_B:
            self.files_B.remove('.ipynb_checkpoints')
        if '.ipynb_checkpoints' in self.labels_files:
            self.labels_files.remove('.ipynb_checkpoints')

    def __getitem__(self, idx):          
        x1 = cv2.imread(os.path.join(self.folder_A, self.files_A[idx]),cv2.IMREAD_GRAYSCALE)
        x2 = cv2.imread(os.path.join(self.folder_B, self.files_B[idx]),cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(self.labels_folder, self.labels_files[idx]),cv2.IMREAD_GRAYSCALE)
        # image=np.array(image).astype("float32")
        # image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        x1=np.array(x1).astype("float32")
        x2=np.array(x2).astype("float32")
        gt=np.array(gt).astype("float32")
        x1 = cv2.resize(x1, (ISIZE, ISIZE))
        x2 = cv2.resize(x2, (ISIZE, ISIZE))
        gt = cv2.resize(gt, (ISIZE, ISIZE))        
        # return image, label
        x1 = np.expand_dims(x1, axis=0)
        x2 = np.expand_dims(x2, axis=0)
        gt = np.expand_dims(gt, axis=0)

        x1 /= 255.0
        x2 /= 255.0
        gt /= 255.0

        return x1, x2, gt

    def __len__(self):
        return self.file_size
    
    def get_files_name(self):
        return self.labels_files
    
# test the custom dataset
if __name__ == "__main__":
     # create a dataset object
    dataset = Custom_dataset(folder_Path=TRAIN_DATA,transform_type="train")
    # create a dataloader object
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # get the first batch
    for x1,x2,gt in dataloader:
        print(x1.shape)
        print(x2.shape)
        print(gt.shape)
        break
    # display the first image
    plb.imshow(x1[0][0])
    plb.show()
    plb.imshow(x2[0][0])
    plb.show()
    plb.imshow(gt[0][0])
    plb.show()
