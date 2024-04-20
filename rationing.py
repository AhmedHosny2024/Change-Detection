import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale images
image1 = cv2.imread('../trainval/trainval/A/0069.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('../trainval/trainval/B/0069.png', cv2.IMREAD_GRAYSCALE)

# Ensure images have the same dimensions
if image1.shape != image2.shape:
    raise ValueError("Images must have the same dimensions")

# Convert images to floating point to avoid overflow during division
image1_float = image1.astype(float)
image2_float = image2.astype(float)

# Perform ratioing
ratio_image = np.divide(image2_float, image1_float, out=np.zeros_like(image2_float), where=image1_float!=0)

# Normalize ratio image for display
ratio_image_normalized = cv2.normalize(ratio_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# Display the ratio image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Image 1')
plt.imshow(image1, cmap='binary')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Ratio Image (Image 2 / Image 1)')
plt.imshow(ratio_image_normalized)  # Using 'jet' colormap for better visualization
plt.colorbar()
plt.axis('off')

plt.show()
