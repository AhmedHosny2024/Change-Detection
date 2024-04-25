import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
def calculate_white_pixel_sum(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Calculate the sum of white pixels
    white_pixel_sum = cv2.countNonZero(image)  # Count non-zero (white) pixels
    return white_pixel_sum

def create_histogram(image_folder):
    # Initialize lists to store image paths and white pixel sums
    white_pixel_sums = np.zeros((256*256))
    print(white_pixel_sums.shape)
    c=0
    # Iterate through each file in the folder
    for filename in os.listdir(image_folder):
        # Check if the file is an image
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Construct the full path to the image
            image_path = os.path.join(image_folder, filename)
            # Calculate the white pixel sum for the image
            white_pixel_sum = calculate_white_pixel_sum(image_path)
            if( white_pixel_sum==1):
                print(image_path)
            # Append the white pixel sum to the list
            white_pixel_sums[white_pixel_sum]+=1
            c+=1
    print(c)

    # Create histogram
    print(len(white_pixel_sums))
    print(white_pixel_sums[0])
    print(white_pixel_sums)
    write_file = open("histogram.txt", "w")
    for i in range(256*256):
        write_file.write(str(white_pixel_sums[i]) + "\n")
    # plt.hist(white_pixel_sums, bins=256*265, color='blue', edgecolor='black', alpha=0.7)
    # plt.xlabel('Sum of White Pixels')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Sum of White Pixels')
    # plt.show()

# Path to the folder containing binary images
image_folder = "trainval/label"

# Create histogram
create_histogram(image_folder)
