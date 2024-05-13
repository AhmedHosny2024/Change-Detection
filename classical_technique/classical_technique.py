import os
import sys
import cv2
import numpy as np
import imutils
from skimage.metrics import structural_similarity as compare_ssim
# from scipy.cluster.vq import kmeans2
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from skimage import exposure

# THRESHOLD=300 0.079
# THRESHOLD = 500  0.06669160304252116
# THRESHOLD = 400 0.08084878138885547
# THRESHOLD= 350  0.080134097312698
THRESHOLD = 190

def preprocess_image(image):
    # Convert image to uint8 data type and ensure it's in the range [0, 255]
    # image = cv2.convertScaleAbs(image, alpha=1.0, beta=0)
    image = (image * 255).astype(np.uint8)
    # # Apply histogram equalization to each color channel separately
    equalized_channels = [exposure.equalize_hist(image[:, :, i]) for i in range(image.shape[2])]
    image = np.stack(equalized_channels, axis=-1)
    # # Apply Gaussian blur
    # image = cv2.GaussianBlur(image, (15, 15), 0)
    # corrected_image = np.copy(image)
    # image=cv2.medianBlur(image, 3)
    image = (image * 255).astype(np.uint8)

    return image
THRESHOLD_SMALL=240
THRESHOLD_LARGE=300
def image_differencing_all_bands(images_A, images_B):
    results_img=[]
    for img_A, img_B in zip(images_A, images_B):
        diff_image = np.abs(img_A - img_B)
        change_mask = np.sum(diff_image, axis=2) 
        change_mask= change_mask>THRESHOLD_SMALL and change_mask<THRESHOLD_LARGE
        change_mask=change_mask.astype(np.uint8)*255
        results_img.append(change_mask)
    return results_img

def image_differencing_YUV(images_A, images_B):
    results_img=[]
    for img_A, img_B in zip(images_A, images_B):
        # change 2 images to binary
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2YUV)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2YUV)
        img_A=img_A[:,:,-1]
        img_B=img_B[:,:,-1]
        THRESHOLD_A= cv2.mean(img_A)[0]
        img_A = cv2.threshold(img_A, THRESHOLD_A, 255, cv2.THRESH_BINARY)[1]
        THRESHOLD_B= cv2.mean(img_B)[0]
        img_B = cv2.threshold(img_B, THRESHOLD_B, 255, cv2.THRESH_BINARY)[1]
        diff_img = cv2.absdiff(img_A, img_B)
        results_img.append(diff_img)
    return results_img

def image_differencing_color(images_A, images_B):
    results_img=[]
    image_number=0
    for img_A, img_B in zip(images_A, images_B):
        # change 2 images to binary
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
        # calculate histogram
        THRESHOLD_A= cv2.mean(img_A)[0]
        img_A = cv2.threshold(img_A, THRESHOLD_A, 255, cv2.THRESH_BINARY)[1]
        THRESHOLD_B= cv2.mean(img_B)[0]
        img_B = cv2.threshold(img_B, THRESHOLD_B, 255, cv2.THRESH_BINARY)[1]
        # Compute the absolute difference between the images
        diff_img = cv2.absdiff(img_A, img_B)
        results_img.append(diff_img)
        # change_mask = np.abs(img_A - img_B)
        # change_mask = change_mask.astype(np.uint8) * 255
        # results_img.append(change_mask)
        # image_number+=1
    return results_img

def image_differencing_gray(images_A, images_B):
    results_img=[]
    for img_A, img_B in zip(images_A, images_B):
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
        diff_image = np.abs(img_A - img_B)
        # THRESHOLD= cv2.mean(diff_image)[0]
        change_mask = diff_image > THRESHOLD
        change_mask=change_mask.astype(np.uint8)*255
        results_img.append(change_mask)
    return results_img

def image_differencing_hue(images_A, images_B):
    results_img=[]
    for img_A, img_B in zip(images_A, images_B):
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2HSV)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2HSV)
        # drop value channel
        img_A=img_A[:,:,0]
        img_B=img_B[:,:,0]
        THRESHOLD_A= cv2.mean(img_A)[0]
        img_A = cv2.threshold(img_A, THRESHOLD_A, 255, cv2.THRESH_BINARY)[1]
        THRESHOLD_B= cv2.mean(img_B)[0]
        img_B = cv2.threshold(img_B, THRESHOLD_B, 255, cv2.THRESH_BINARY)[1]
        diff_img = cv2.absdiff(img_A, img_B)
        results_img.append(diff_img)
    return results_img 

def image_differencing(images_A, images_B): 
    results_img=[]
    for img_A, img_B in zip(images_A, images_B):
        # Convert images to grayscale
        gray_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
        gray_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
        # Compute the absolute difference between the images
        diff_image = cv2.absdiff(gray_A, gray_B)
        # Apply a threshold to the difference image
        _, change_mask = cv2.threshold(diff_image, THRESHOLD, 255, cv2.THRESH_BINARY)
        results_img.append(change_mask)
    return results_img

# K=5: 0.06393759768844175
# K=8: 0.06593848687200131
# k=20 0.0675898079952895
n_clusters = 2
spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42,max_iter=100)

def post_classification_comparison(images_A, images_B):
    results_img=[]
    image_number=0
    for img_A, img_B in zip(images_A, images_B):
        # Flatten images for classification
        print("Processing image", image_number)
        # img_A = preprocess_image(img_A)
        # img_B = preprocess_image(img_B)
        # Classify images
        classified_image1 = classify_image(img_A)
        classified_image2 = classify_image(img_B)
        # Compare predicted classes to identify changes
        change_map = (classified_image1 != classified_image2).astype(np.uint8) * 255
        change_map = change_map.reshape((256, 256))
        results_img.append(change_map)
        image_number+=1
    return results_img

def classify_image(image):
    X = image.reshape(-1, 3)
    # Initialize and fit K-means clustering algorithm
    kmeans.fit(X)
    # Get the labels assigned to each pixel
    labels = kmeans.labels_
    # Reshape the labels to the original image shape
    classified_image = labels.reshape(image.shape[:2])
    return classified_image

def stronge_classifier(image):
    #  Flatten the image
    X = image.reshape(-1, image.shape[-1])
    labels = spectral_clustering.fit_predict(X)
    classified_image = labels.reshape(image.shape[:2])
    return classified_image

def diff_image(images_A, images_B):
    results_img=[]
    for img_A, img_B in zip(images_A, images_B):
        img_diff = np.array(img_A, dtype=np.float32) - np.array(img_B, dtype=np.float32)
        img_diff = np.sqrt(np.sum(np.square(img_diff), axis=-1))
        img_diff= img_diff>60
        img_diff=img_diff.astype(np.uint8)*255
        results_img.append(img_diff)
    return results_img
# Mean Jaccard Index: 0.09206747368369564
# Mean Jaccard Index: 0.1028530606521822 78
def diff_image_simple(images_A, images_B):
    results_img=[]
    for img_A, img_B in zip(images_A, images_B):
        img_A=preprocess_image(img_A)
        img_B=preprocess_image(img_B)
        img_diff = np.array(img_A, dtype=np.float32) - np.array(img_B, dtype=np.float32)
        img_diff= img_diff>75
        img_diff=img_diff.astype(np.uint8)*255
        results_img.append(img_diff)
    return results_img