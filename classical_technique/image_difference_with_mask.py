import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as compare_ssim

def segment_buildings(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply adaptive thresholding to binarize the image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations to enhance and clean the binary image
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)

    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask to store the segmented buildings
    mask = np.zeros_like(gray)
    
    # Iterate through the contours and draw them on the mask
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        if area >800 and area<10000:  # Adjust this threshold according to your needs
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    
    return mask


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

        # Segment buildings in both images
        building_mask_A = segment_buildings(img_A)
        building_mask_B = segment_buildings(img_B)
        
        diff_img = cv2.absdiff(building_mask_B, building_mask_A)

        # Save the difference image
        output_path = os.path.join(output_folder, file_A.split('.')[0] + '_diff.png')
        cv2.imwrite(output_path, diff_img)

def compute_jaccard_index(result_folder, label_folder):
    # This function remains the same
    pass

# Example usage
folder_A = "trainval/A-test"
folder_B = "trainval/B-test"
output_folder = "difference_images"
label_folder = "trainval/label-test"

image_differencing(folder_A, folder_B, output_folder)
# compute_jaccard_index(output_folder, label_folder)
