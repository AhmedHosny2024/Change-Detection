# normlise
import os
import sys
import numpy as np
import cv2
from skimage import exposure
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def preprocess_remote_sensing_data(image,calibrated=True,
                                   enhancement=True, noise_reduction=True, dropout_prob=True):
    
    if calibrated:
        image = cv2.convertScaleAbs(image, alpha=1.0, beta=0)
    # Perform image enhancement
    if enhancement:
        # Histogram equalization for enhancing contrast
        image = exposure.equalize_hist(image)

    # Perform noise reduction
    if noise_reduction:
        # Gaussian filtering for noise reduction
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # Perform line dropout
    if dropout_prob:
        # Randomly drop lines in the image
        image=locate_and_correct_bad_lines(image)

    return image

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
    return image

def locate_and_correct_bad_lines(image):
    # Iterate over each row in the image
    corrected_image = np.copy(image)
    for i in range(1, image.shape[0]-1):
        # Find the locations of white pixels (255) in the current row
        # dropout_locations = np.where(image[i] == 255)[0]        
        # Iterate over each dropout location in the current row
        if np.sum(image[i] == 255) > 255 or np.sum(image[i] == 0) > 255:
            # Get the pixel values from the preceding and succeeding lines
            # print("Line",image[i])
            prev_value = image[i-1]
            next_value = image[i+1]
            # print("in loop")
            # Calculate the average of the two values
            average_value = (prev_value + next_value) // 2
            # print("average_value",average_value)
            # Assign the average value to the dropout pixel
            corrected_image[i] = average_value
    
    return corrected_image
if __name__ == "__main__":
    # Input and output folders
    input_folder = 'trainval/B'
    output_folder = 'trainval/B_preprocessed'

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through images in input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Read image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            # Preprocess image
            preprocessed_image = preprocess_image(image)
            # Save preprocessed image
            # show the image
            # preprocessed_image=preprocessed_image*255
            preprocessed_image = (preprocessed_image * 255).astype(np.uint8)

            # preprocessed_image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, preprocessed_image)

            print(f"Processed and saved: {output_path}")
            break

    print("All images processed and saved successfully.")




