import os
import sys
import cv2
import numpy as np
import imutils
from skimage.metrics import structural_similarity as compare_ssim
from scipy.cluster.vq import kmeans2

# THRESHOLD=300 0.079
# THRESHOLD = 500  0.06669160304252116
# THRESHOLD = 400 0.08084878138885547
# THRESHOLD= 350  0.080134097312698
THRESHOLD = 400
def image_differencing(images_A, images_B):
    results_img=[]
    for img_A, img_B in zip(images_A, images_B):
        diff_image = np.abs(img_A - img_B)
        change_mask = np.sum(diff_image, axis=2) > THRESHOLD
        change_mask=change_mask.astype(np.uint8)*255
        results_img.append(change_mask)
    return results_img
# K=5: 0.06393759768844175
# K=8: 0.06593848687200131
# k=20 0.0675898079952895
def post_classification_comparison(images_A, images_B):
    results_img=[]
    for img_A, img_B in zip(images_A, images_B):
        # Flatten images for classification
        X1 = img_A.reshape(-1, 3).astype(float)  # Convert to float
        X2 = img_B.reshape(-1, 3).astype(float)  # Convert to float
        # Perform K-means clustering on each image
        centroids1, labels1 = kmeans2(X1, k=20)
        labels1 = labels1.reshape((256, 256))

        centroids2, labels2 = kmeans2(X2, k=20)
        labels2 = labels2.reshape((256, 256))

        # Compare predicted classes to identify changes
        change_map = (labels1 != labels2).astype(np.uint8) * 255
        results_img.append(change_map)
    return results_img



    

