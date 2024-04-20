import cv2
import numpy as np

# Load images
image1 = cv2.imread('trainval/B-test/0069.png')
image2 = cv2.imread('trainval/A-test/0069.png')

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Compute absolute difference
diff = cv2.absdiff(gray1, gray2)

# Threshold the difference image
_, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Perform morphological operations
kernel = np.ones((5, 5), np.uint8)
thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
result = image1.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
