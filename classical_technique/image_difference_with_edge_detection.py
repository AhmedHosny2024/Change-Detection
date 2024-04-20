import cv2
import numpy as np

# Read the remote sensing image
image = cv2.imread('trainval/B/0118.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Perform edge detection using Canny edge detector
edges = cv2.Canny(blurred, 50, 150)

# Perform dilation to connect nearby edges
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=2)
# Find contours of connected components
contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on approximate shape (rectangle)
filtered_contours = []
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.1 * perimeter, True)
    if len(approx) == 4:  # Check if contour is approximately rectangular
        filtered_contours.append(contour)

# Create an empty binary image
binary = np.zeros_like(gray)

# Draw filtered contours on binary image
cv2.drawContours(binary, filtered_contours, -1, 255, thickness=cv2.FILLED)

# Display the binary image
cv2.imshow('Binary Image with Filtered Contours', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
