import os
import shutil
import random
import cv2

def split_dataset(data_dir, train_dir, test_dir, val_dir, split_ratio=(0.75, 0.125, 0.125)):
    assert sum(split_ratio) == 1.0, "Split ratio should sum up to 1.0"
    
    # Create directories if they don't exist
    for folder in [train_dir, test_dir, val_dir]:
        os.makedirs(folder, exist_ok=True)
    
    # Get list of files in A folder
    files_A = os.listdir(os.path.join(data_dir, 'A'))
    num_files_A = len(files_A)
    
    # Get list of files in B folder
    files_B = os.listdir(os.path.join(data_dir, 'B'))
    num_files_B = len(files_B)
    
    # Get list of files in Label folder
    files_Label = os.listdir(os.path.join(data_dir, 'Label'))
    num_files_Label = len(files_Label)



    
    # Make sure all folders have the same number of files
    assert num_files_A == num_files_B == num_files_Label, "Number of files in folders A, B, and Label must be the same"
    
    # Shuffle indices of files
    indices = list(range(num_files_A))
    random.shuffle(indices)
    
    # Split indices into train, test, and validation sets
    train_split = int(num_files_A * split_ratio[0])
    test_split = int(num_files_A * split_ratio[1])


    train_indices = indices[:train_split]
    test_indices = indices[train_split:train_split + test_split]
    val_indices = indices[train_split + test_split:]

    # Copy files to train, test, and validation folders
    for idx in train_indices:
        filename_A = files_A[idx]
        filename_B = files_B[idx]
        filename_Label = files_Label[idx]
        shutil.copy(os.path.join(data_dir, 'A', filename_A), os.path.join(train_dir, 'A', filename_A))
        shutil.copy(os.path.join(data_dir, 'B', filename_B), os.path.join(train_dir, 'B', filename_B))
        shutil.copy(os.path.join(data_dir, 'Label', filename_Label), os.path.join(train_dir, 'Label', filename_Label))
    
    for idx in test_indices:
        filename_A = files_A[idx]
        filename_B = files_B[idx]
        filename_Label = files_Label[idx]
        
        shutil.copy(os.path.join(data_dir, 'A', filename_A), os.path.join(test_dir, 'A', filename_A))
        shutil.copy(os.path.join(data_dir, 'B', filename_B), os.path.join(test_dir, 'B', filename_B))
        shutil.copy(os.path.join(data_dir, 'Label', filename_Label), os.path.join(test_dir, 'Label', filename_Label))
    
    for idx in val_indices:
        filename_A = files_A[idx]
        filename_B = files_B[idx]
        filename_Label = files_Label[idx]
        
        shutil.copy(os.path.join(data_dir, 'A', filename_A), os.path.join(val_dir, 'A', filename_A))
        shutil.copy(os.path.join(data_dir, 'B', filename_B), os.path.join(val_dir, 'B', filename_B))
        shutil.copy(os.path.join(data_dir, 'Label', filename_Label), os.path.join(val_dir, 'Label', filename_Label))
def calculate_white_pixel_sum(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Calculate the sum of white pixels
    white_pixel_sum = cv2.countNonZero(image)  # Count non-zero (white) pixels
    return white_pixel_sum

def split_dataset_by_balancing():
    count_one_train=0
    count_zero_train=0
    count_zero_test=0
    count_one_test=0
    count_zero_val=0
    count_one_val=0
    image_folder = "trainval/Label"
    files_Label = os.listdir(image_folder)
    num_files_Label = len(files_Label)
    print(num_files_Label)
    indices = list(range(num_files_Label))
    random.shuffle(indices)
    for idx in indices:
    # Check if the file is an image
        if files_Label[idx].endswith(".png") or files_Label[idx].endswith(".jpg"):
            # Construct the full path to the image
            image_path_label = os.path.join(image_folder, files_Label[idx])
            image_path_A=os.path.join("trainval/A",files_Label[idx])
            image_path_B=os.path.join("trainval/B",files_Label[idx])
            # Calculate the white pixel sum for the image
            white_pixel_sum = calculate_white_pixel_sum(image_path_label)
            if white_pixel_sum>500:
                if count_one_train<1200:
                    shutil.copy(image_path_label, os.path.join("trainval/train_balanced/Label",files_Label[idx]))
                    shutil.copy(image_path_A, os.path.join("trainval/train_balanced/A",files_Label[idx]))
                    shutil.copy(image_path_B, os.path.join("trainval/train_balanced/B",files_Label[idx]))
                    count_one_train+=1
                elif count_one_test<218:
                    shutil.copy(image_path_label, os.path.join("trainval/test_balanced/Label",files_Label[idx]))
                    shutil.copy(image_path_A, os.path.join("trainval/test_balanced/A",files_Label[idx]))
                    shutil.copy(image_path_B, os.path.join("trainval/test_balanced/B",files_Label[idx]))
                    count_one_test+=1
                elif count_one_val<218:
                    shutil.copy(image_path_label, os.path.join("trainval/validation_balanced/Label",files_Label[idx]))
                    shutil.copy(image_path_A, os.path.join("trainval/validation/A",files_Label[idx]))
                    shutil.copy(image_path_B, os.path.join("trainval/validation/B",files_Label[idx]))
                    count_one_val+=1
            else:
                if count_zero_train<1200:
                    shutil.copy(image_path_label, os.path.join("trainval/train_balanced/Label",files_Label[idx]))
                    shutil.copy(image_path_A, os.path.join("trainval/train_balanced/A",files_Label[idx]))
                    shutil.copy(image_path_B, os.path.join("trainval/train_balanced/B",files_Label[idx]))
                    count_zero_train+=1
                elif count_zero_test<1016:
                    shutil.copy(image_path_label, os.path.join("trainval/test_balanced/Label",files_Label[idx]))
                    shutil.copy(image_path_A, os.path.join("trainval/test_balanced/A",files_Label[idx]))
                    shutil.copy(image_path_B, os.path.join("trainval/test_balanced/B",files_Label[idx]))
                    count_zero_test+=1
                elif count_zero_val<1016:
                    shutil.copy(image_path_label, os.path.join("trainval/validation_balanced/Label",files_Label[idx]))
                    shutil.copy(image_path_A, os.path.join("trainval/validation_balanced/A",files_Label[idx]))
                    shutil.copy(image_path_B, os.path.join("trainval/validation_balanced/B",files_Label[idx]))
                    count_zero_val+=1

def get_only_changes():
    image_folder = "trainval/validation/Label"
    files_Label = os.listdir(image_folder)
    num_files_Label = len(files_Label)
    indices = list(range(num_files_Label))
    # random.shuffle(indices)
    for idx in indices:
    # Check if the file is an image
        if files_Label[idx].endswith(".png") or files_Label[idx].endswith(".jpg"):
            # Construct the full path to the image
            image_path_label = os.path.join(image_folder, files_Label[idx])
            image_path_A=os.path.join("trainval/validation/A",files_Label[idx])
            image_path_B=os.path.join("trainval/validation/B",files_Label[idx])
            
            # Calculate the white pixel sum for the image
            white_pixel_sum = calculate_white_pixel_sum(image_path_label)
            if white_pixel_sum==0:
            # if  path fiounf in A B Label then copy to new folder
                if os.path.exists(image_path_A) and os.path.exists(image_path_B):
                    shutil.copy(image_path_label, os.path.join("trainval/test_no_change_data/Label",files_Label[idx]))
                    shutil.copy(image_path_A, os.path.join("trainval/test_no_change_data/A",files_Label[idx]))
                    shutil.copy(image_path_B, os.path.join("trainval/test_no_change_data/B",files_Label[idx]))
def make_validation_folder_balanced():
    c=0
    image_folder = "trainval/A"
    files_A = os.listdir(image_folder)
    num_files_A = len(files_A)
    indices = list(range(num_files_A))
    print(num_files_A)
    # random.shuffle(indices)

    for idx in indices:
    # Check if the file is an image
        if files_A[idx].endswith(".png") or files_A[idx].endswith(".jpg"):
            # Construct the full path to the image
            image_path_A=os.path.join("trainval/A",files_A[idx])
            image_path_B=os.path.join("trainval/B",files_A[idx])
            image_path_label=os.path.join("trainval/label",files_A[idx])
            image_test_check=os.path.join("trainval/test_onserver/A",files_A[idx])
            image_validation_check=os.path.join("trainval/validation_onsever/A",files_A[idx])
            # if  path fiounf in A B Label then copy to new folder
            if not os.path.exists(image_test_check) and not os.path.exists(image_validation_check):
                shutil.copy(image_path_label, os.path.join("trainval/train_onserver/A",files_A[idx]))
                shutil.copy(image_path_A, os.path.join("trainval/train_onserver/B",files_A[idx]))
                shutil.copy(image_path_B, os.path.join("trainval/train_onserver/label",files_A[idx]))
                c+=1
    print(c)
    
# Example usage
# data_dir = 'trainval'
# train_dir = 'trainval/train'
# test_dir = 'trainval/test'
# val_dir = 'trainval/validation'

# split_dataset(data_dir, train_dir, test_dir, val_dir,split_ratio=(0.75, 0.125, 0.125))
# split_dataset_by_balancing()
# get_only_changes()
# make_validation_folder_balanced()
