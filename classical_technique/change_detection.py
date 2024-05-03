import os
import numpy as np
import cv2
from classical_technique.custom_data_loader import CustomDataset
from sklearn.metrics import jaccard_score
from classical_technique.classical_technique import *

class ChangeDetection:
    def __init__(self, folder_A, folder_B,output_folder,labels_folder,model=None):
        self.folder_A = folder_A
        self.folder_B = folder_B
        self.output_folder = output_folder
        self.labels_folder = labels_folder
        self.model = model
        self.custom_dataset = CustomDataset(self.folder_A, self.folder_B)
        self.images_A,self.images_B=self.custom_dataset.data_loader()

    def train(self):
        print("Training the model...")
        # Perform training callings
        results=[]
        results = image_differencing(self.images_A, self.images_B)
        # results = post_classification_comparison(self.images_A, self.images_B)
        for i, out_image in enumerate(results):
        # Save the difference image
            filename = f"{i:04d}.png"
            output_path = os.path.join(self.output_folder, filename)
            cv2.imwrite(output_path, out_image)

    def evaluate(self):
        print("Evaluating the model...")
        # Perform evaluation callings
        if not os.path.exists(self.output_folder) or not os.path.exists(self.labels_folder):
            print("Result or label folder does not exist.")
            return
        # Get list of files in result folder
        result_files = sorted(os.listdir(self.output_folder))
        jaccard_indices = []
        for result_file in result_files:
            # Read result and label images
            result_img = cv2.imread(os.path.join(self.output_folder, result_file))
            label_img = cv2.imread(os.path.join(self.labels_folder, result_file))
            # Compute Jaccard Index
            img_true=np.array(label_img).ravel()
            img_pred=np.array(result_img).ravel()
            jaccard_index = jaccard_score(img_true, img_pred,pos_label=1,zero_division=1)
            jaccard_indices.append(jaccard_index)
        mean_jaccard_index = np.mean(jaccard_indices)
        print("Mean Jaccard Index:", mean_jaccard_index)
# main
if __name__ == "__main__":
    folder_A = "trainval/A"
    folder_B = "trainval/B"
    output_folder = "trainval/difference_images"
    labels_folder = "trainval/label"
    change_detection = ChangeDetection(folder_A, folder_B,output_folder,labels_folder)
    change_detection.train()
    change_detection.evaluate()
