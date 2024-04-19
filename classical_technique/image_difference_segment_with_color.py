import cv2
import numpy as np

# Read the remote sensing image
image = cv2.imread('trainval/B-test/3882.png')

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for the white color (can adjust these values)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])

# Threshold the HSV image to get only white colors
mask = cv2.inRange(hsv, lower_white, upper_white)

# Apply morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Invert the mask
sure_fg = cv2.bitwise_not(sure_bg)

# Combine the foreground and background
segmented = cv2.bitwise_and(image, image, mask=sure_fg)

# Display the segmented image
cv2.imshow('Segmented Image', segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()
