import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import jaccard_score


def segment_buildings(image,kernel_size=(5,5),iterations=2,area_ranges=(800,10000),gaussian=15,):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (gaussian, gaussian), 0)
    
    # Apply adaptive thresholding to binarize the image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations to enhance and clean the binary image
    kernel = np.ones(kernel_size, np.uint8)
    
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)

    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask to store the segmented buildings
    mask = np.zeros_like(gray)
    
    # Iterate through the contours and draw them on the mask
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >area_ranges[0] and area<area_ranges[1]:  # Adjust this threshold according to your needs
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    
    return mask


def image_differencing(folder_A, folder_B, output_folder,kernel_size=5,iterations=2,area_ranges=(800,10000),gaussian=15,):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of files in both folders
    files_A = sorted(os.listdir(folder_A))
    files_B = sorted(os.listdir(folder_B))

    # Check if the number of images in both folders are same
    if len(files_A) != len(files_B):
        print("Number of images in folder A and B should be same.")
        return

    # i=0
    for file_A, file_B in zip(files_A, files_B):
        # Read images
        img_A = cv2.imread(os.path.join(folder_A, file_A))
        img_B = cv2.imread(os.path.join(folder_B, file_B))

        # Segment buildings in both images
        building_mask_A = segment_buildings(img_A,kernel_size,iterations,area_ranges,gaussian)
        building_mask_B = segment_buildings(img_B,kernel_size,iterations,area_ranges,gaussian)
        
       
        diff_img = cv2.absdiff(building_mask_B, building_mask_A)

        # Save the difference image
        output_path = os.path.join(output_folder, file_A.split('.')[0] + '_diff.png')
        cv2.imwrite(output_path, diff_img)

        # if i==15:
        #   # show the segmented images
        #   cv2.imshow('building_mask_A', building_mask_A)
        #   cv2.waitKey(0)
        #   cv2.imshow('building_mask_B', building_mask_B)
        #   cv2.waitKey(0)
        #   break
        # i+=1

def compute_jaccard_index(result_folder, label_folder):
    # This function remains the same
    
    result_files = sorted(os.listdir(result_folder))
    labeled_files = sorted(os.listdir(label_folder))
    jaccard_indices = []

    for result_file,label_file in zip(result_files,labeled_files):
      # Read result and label images
      result_img = cv2.imread(os.path.join(result_folder, result_file))
      label_img = cv2.imread(os.path.join(label_folder, label_file))
      # Compute Jaccard Index
      img_true=np.array(label_img).ravel()
      img_pred=np.array(result_img).ravel()
      jaccard_index = jaccard_score(img_true, img_pred,pos_label=255)
      jaccard_indices.append(jaccard_index)
    mean_jaccard_index = np.mean(jaccard_indices)
    print("Mean Jaccard Index:", mean_jaccard_index)
    return mean_jaccard_index

# Example usage
folder_A = "trainval/A"
folder_B = "trainval/B"
output_folder = "difference_images"
label_folder = "trainval/label"

# image_differencing(folder_A, folder_B, output_folder)
# compute_jaccard_index(output_folder, label_folder)


def tune_parameters(folder_A, folder_B):
    # Define the range of parameters to try
    kernel_sizes = [(3, 3), (5, 5), (7, 7)]
    iterations = [1, 2, 3]
    area_ranges = [(500, 15000), (800, 10000), (1000, 8000)]

    best_score = 0
    best_params = None

    # Try all combinations of parameters
    i=0
    for kernel_size in kernel_sizes:
        for iteration in iterations:
            for area_range in area_ranges:
                print("start number ",i)
                i+=1
                # Run the image differencing function with the current parameters
                image_differencing(folder_A, folder_B, output_folder, kernel_size, iteration, area_range)

                # Compute the Jaccard index for the current parameters
                score = compute_jaccard_index(output_folder, label_folder)

                # If the current score is better than the best score, update the best score and best parameters
                if score > best_score:
                    best_score = score
                    best_params = (kernel_size, iteration, area_range)

    return best_params

tune_parameters(folder_A,folder_B)