
import numpy as np
import os
import cv2
class CustomDataset:
    def __init__(self,folder_A, folder_B):
        self.folder_A=folder_A
        self.folder_B=folder_B
        self.files_A = sorted(os.listdir(folder_A))
        self.files_B = sorted(os.listdir(folder_B))
        self.images_A = []
        self.images_B = []

        # Check if the number of images in both folders are same
        if len(self.files_A) != len(self.files_B):
            print("Number of images in folder A and B should be same.",len(self.files_A),len(self.files_B))
            print("Number of images in folder A and B should be same.")
            return

    def data_loader(self):
        for file_A, file_B in zip(self.files_A, self.files_B):
            img_A = cv2.imread(os.path.join(self.folder_A, file_A))
            img_B = cv2.imread(os.path.join(self.folder_B, file_B))
            self.images_A.append(img_A)
            self.images_B.append(img_B)
        return self.images_A,self.images_B
