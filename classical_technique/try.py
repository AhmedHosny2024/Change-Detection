import numpy as np
import cv2

# Load the two RGB images
image1 = cv2.imread('trainval/A/0118.png')
image2 = cv2.imread('trainval/B/0118.png')

# Convert images to float32 for calculations
image1 = image1.astype(np.float32)
image2 = image2.astype(np.float32)

# Calculate the absolute difference between the two images
diff_image = np.abs(image1 - image2)
print(diff_image.shape)

# Define a threshold for change detection
threshold = 260  # You can adjust this threshold as needed

# Create a binary mask where pixels above the threshold are considered changed
change_mask = np.sum(diff_image, axis=2) > threshold
print(change_mask.shape)
print(change_mask)
change_mask=change_mask.astype(np.uint8)*255
print(change_mask)
# # Apply the change mask to the original images
# changed_pixels = image2 * change_mask[..., np.newaxis]
# unchanged_pixels = image2 * ~change_mask[..., np.newaxis]

# Save the result images
cv2.imwrite('mask_change.jpg', change_mask)