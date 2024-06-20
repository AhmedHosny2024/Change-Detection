import os
import numpy as np
import cv2
from classical.custom_data_loader import CustomDataset
from sklearn.metrics import jaccard_score
from classical.classical_technique import *
# from classical_technique.PCAKMeans import *
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ChangeDetection:
    def __init__(self, folder_A, folder_B,output_folder,labels_folder,model=None):
        self.folder_A = folder_A
        self.folder_B = folder_B
        self.output_folder = output_folder
        self.labels_folder = labels_folder
        self.model = model
        self.custom_dataset = CustomDataset(self.folder_A, self.folder_B)
        self.images_A,self.images_B=self.custom_dataset.data_loader()
        self.THRESHOLD=0.5

    def train(self):
        print("Training the model...")
        # Perform training callings
        results=[]
        # results = post_classification_comparison(self.images_A, self.images_B)
        # results = post_classification_comparison(self.images_A, self.images_B)
        # results=pca_kmeans_change(self.images_A, self.images_B)    
        results=diff_image_simple(self.images_A, self.images_B)
        # results=cva_change(self.images_A, self.images_B)
        for i, out_image in enumerate(results):
        # Save the difference image
            filename = f"{i:04d}.png"
            output_path = os.path.join(self.output_folder, filename)
            cv2.imwrite(output_path, out_image)
    def f1_score(self,score, lb):
        threshold =self.THRESHOLD
        score[score>threshold] = 1.0
        score[score<=threshold] = 0.0 
        lb[lb>threshold] = 1.0
        lb[lb<=threshold] = 0.0 
        lb = lb[0,:,:]
        lb = np.round(lb)
        tp = np.sum(lb*score)
        fn = lb-score
        fn[fn<0]=0
        fn = np.sum(fn)
        tn = lb+score
        tn[tn>0]=-1
        tn[tn>=0]=1
        tn[tn<0]=0
        tn = np.sum(tn)
        fp = score - lb
        fp[fp<0] = 0
        fp = np.sum(fp)
        return tp, fp, tn, fn
    def evaluate(self):
        print("Evaluating the model...")
        # Perform evaluation callings
        if not os.path.exists(self.output_folder) or not os.path.exists(self.labels_folder):
            print("Result or label folder does not exist.")
            return
        # Get list of files in result folder
        result_files = sorted(os.listdir(self.output_folder))
        label_files = sorted(os.listdir(self.labels_folder))
        jaccard_indices = []
        Tp=0
        Fp=0
        Tn=0
        Fn=0
        labels=[]
        results=[]
        for result_file,label_file in zip(result_files,label_files):
            # Read result and label images
            result_img = cv2.imread(os.path.join(self.output_folder, result_file))
            label_img = cv2.imread(os.path.join(self.labels_folder, label_file))
            labels.append(label_img.tolist())
            results.append(result_img.tolist())
            # take copy
            result_img_copy = result_img.copy()
            result_img_copy=result_img_copy/255
            label_img_copy = label_img.copy()
            label_img_copy=label_img_copy/255
            # f1 score
            tp, fp, tn, fn = self.f1_score(result_img_copy, label_img_copy)
            Tp+=tp
            Fp+=fp
            Tn+=tn
            Fn+=fn
            # Compute Jaccard Index
            img_true=np.array(label_img).ravel()
            img_pred=np.array(result_img).ravel()
            jaccard_index = jaccard_score(img_true, img_pred,pos_label=255,zero_division=1)
            jaccard_indices.append(jaccard_index)
        mean_jaccard_index = np.mean(jaccard_indices)

        precision = Tp/(Tp+Fp)
        recall = Tp/(Tp+Fn)
        f1_score = 2*precision*recall/(precision+recall)
        print("Confusion Matrix:")
        print("Total:", Tp+Fp+Tn+Fn)
        print("Accuracy:", (Tp+Tn)/(Tp+Fp+Tn+Fn))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1_score)
        print("Mean Jaccard Index:", mean_jaccard_index)
        print("Confusion Matrix:")
        conf_matrix = np.array([[Tn, Fp], [Fn, Tp]])
        # Plot confusion matrix
        plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
        plt.title('Confusion Matrix')
        plt.colorbar()
        # Add labels
        plt.xticks([0, 1], ['Predicted Negative', 'Predicted Positive'])
        plt.yticks([0, 1], ['Actual Negative', 'Actual Positive'])
        # Add text annotations
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', verticalalignment='center', color='black')

        plt.show()
    
# main
if __name__ == "__main__":
    folder_A = "trainval/A"
    folder_B = "trainval/B"
    output_folder = "trainval/difference_images"
    labels_folder = "trainval/label"
    change_detection = ChangeDetection(folder_A, folder_B,output_folder,labels_folder)
    change_detection.train()
    change_detection.evaluate()
