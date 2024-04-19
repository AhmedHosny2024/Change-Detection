import os
import sys
import cv2
import numpy as np
import imutils
from skimage.metrics import structural_similarity as compare_ssim


def image_differencing(folder_A, folder_B, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of files in both folders
    files_A = sorted(os.listdir(folder_A))
    files_B = sorted(os.listdir(folder_B))

    # Check if the number of images in both folders are same
    if len(files_A) != len(files_B):
        print("Number of images in folder A and B should be same.")
        return

    for file_A, file_B in zip(files_A, files_B):
        # Read images
        img_A = cv2.imread(os.path.join(folder_A, file_A))
        img_B = cv2.imread(os.path.join(folder_B, file_B))

        # convert the images to grayscale
        grayA = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)

        # Perform image differencing
        (score, diff_img) = compare_ssim(grayA, grayB, full=True)
        diff_img = (diff_img*255).astype("uint8")

        # Apply threshold. Apply both THRESH_BINARY_INV and THRESH_OTSU
        diff_thresh = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # extract the contours from the thresholded image
        contours= cv2.findContours(diff_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = imutils.grab_contours(contours)

        mask = np.zeros_like(grayA)
    
        # Iterate through the contours and draw them on the mask
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >1000 and area<8000:  # Adjust this threshold according to your needs
                x, y, w, h = cv2.boundingRect(contour)
                # Draw rectangle - bounding box on both images
                cv2.rectangle(img_A, (x, y), (x+w, y+h), (0,0,255), 2)
                cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

        # Save the difference image
        output_path = os.path.join(output_folder, file_A.split('.')[0] + '_diff.png')
        cv2.imwrite(output_path, mask)
        cv2.imshow("Differences", img_A)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def compute_jaccard_index(result_folder, label_folder):
    if not os.path.exists(result_folder) or not os.path.exists(label_folder):
        print("Result or label folder does not exist.")
        return

    # Get list of files in result folder
    result_files = sorted(os.listdir(result_folder))

    jaccard_indices = []
    for result_file in result_files:
        # Read result and label images
        result_img = cv2.imread(os.path.join(result_folder, result_file), cv2.IMREAD_GRAYSCALE)
        label_img = cv2.imread(os.path.join(label_folder, result_file))
        print(result_img, label_img)
        # Compute Jaccard Index
        intersection = np.logical_and(result_img, label_img)
        union = np.logical_or(result_img, label_img)
        jaccard_index = np.sum(intersection) / np.sum(union)
        jaccard_indices.append(jaccard_index)

    mean_jaccard_index = np.mean(jaccard_indices)
    print("Mean Jaccard Index:", mean_jaccard_index)

# Example usage
folder_A = "trainval\A-test"
folder_B = "trainval\B-test"
output_folder = "difference_images"
label_folder = "trainval\label-test"

image_differencing(folder_B, folder_A, output_folder)
# compute_jaccard_index(output_folder, label_folder)
