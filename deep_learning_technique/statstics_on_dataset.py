import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def calculate_white_pixel_sum(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Calculate the sum of white pixels
    white_pixel_sum = cv2.countNonZero(image)  # Count non-zero (white) pixels
    return white_pixel_sum

def create_histogram(image_folder):
    # Initialize lists to store image paths and white pixel sums
    white_pixel_sums = np.zeros((256*256))
    c=0
    zeros_image=0
    # Iterate through each file in the folder
    for filename in os.listdir(image_folder):
        # Check if the file is an image
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Construct the full path to the image
            image_path_label = os.path.join(image_folder, filename)
            image_path_A=os.path.join("trainval/train_80_removing/A",filename)
            image_path_B=os.path.join("trainval/train_80_removing/B",filename)
            # Calculate the white pixel sum for the image
            white_pixel_sum = calculate_white_pixel_sum(image_path_label)
            # if white_pixel_sum==0 and zeros_image<200:
            #     os.remove(image_path_label)
            #     os.remove(image_path_A)
            #     os.remove(image_path_B)
            #     zeros_image+=1
            # Append the white pixel sum to the list
            white_pixel_sums[white_pixel_sum]+=1
            c+=1
    print(c)
    print(zeros_image)
    # Create histogram
    print(white_pixel_sums[0])
    write_file = open("histogram.txt", "w")
    for i in range(256*256):
        write_file.write(str(white_pixel_sums[i]) + "\n")


def calculate_mean_and_std(folder_path):
    # Initialize variables to accumulate sum of pixel values for each channel
    mean = [0.0, 0.0, 0.0]
    std = [0.0, 0.0, 0.0]
    num_images = 0

    # Iterate through images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(folder_path, filename)).convert('RGB')
            img_np = np.array(img) / 255.0  # Convert to numpy array and normalize
            mean += np.mean(img_np, axis=(0, 1))
            std += np.std(img_np, axis=(0, 1))
            num_images += 1

    # Calculate mean and std
    mean /= num_images
    std /= num_images
    print("num_images",num_images)
    print("mean",mean)
    print("std",std)
    return mean, std


if __name__ == "__main__":

    # Paths to your image folders
    # folder1_path = 'trainval/A'
    # folder2_path = 'trainval/B'
    filepath = "trainval/B"  # Dataset directory
    pathDir = os.listdir(filepath)  # Images in dataset directory
    num = len(pathDir)  # Here (512512) is the size of each image

    print("Computing mean...")
    data_mean = np.zeros(3)
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = Image.open(os.path.join(filepath, filename))
        img = np.array(img) / 255.0
        data_mean += np.mean(img)  # Take all the data of the first dimension in the three-dimensional matrix
		# As the use of gray images, so calculate a channel on it
    data_mean = data_mean / num

    print("Computing var...")
    data_std = 0.
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = Image.open(os.path.join(filepath, filename)).convert('L').resize((256, 256))
        img = np.array(img) / 255.0
        data_std += np.std(img)

    data_std = data_std / num
    print("mean:{}".format(data_mean))
    print("std:{}".format(data_std))
    
#   Mean: [0.78178327 0.7076608  0.61143379]
#   Std: [0.08119502 0.08063477 0.08059688]
#   ================================================================================    #
#   A:
#   mean:[0.69453992 0.69453992 0.69453992]
#   std:0.06241417633736924
#   B:
#   mean:[0.70604532 0.70604532 0.70604532]
#   std:0.09685950890158737
#   ================================================================================    #
#   4868
#   3232
#   2562
#   1800 1
#   1323 0
