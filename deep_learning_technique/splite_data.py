import os
import shutil
import random

def split_dataset(data_dir, train_dir, test_dir, val_dir, split_ratio=(0.8, 0.1, 0.1)):
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

# Example usage
data_dir = 'trainval'
train_dir = 'trainval/train'
test_dir = 'trainval/test'
val_dir = 'trainval/validation'

split_dataset(data_dir, train_dir, test_dir, val_dir, split_ratio=(0.8, 0.1, 0.1))
