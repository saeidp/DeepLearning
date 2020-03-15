import cv2
import numpy as np
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

# RGB to Gray
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

# Gaussian Blur (to reduce nise and smooth image)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection - Canny image
# It computes the gradient in all directions of x and y
# it traces the edge with large chabge in intensity
# (large gradiet) in an outline of white pixels
# If the gradient is bigger than the upper threshold then it
# is accepted as a an edge pixel. If it is below the lower
#  threshold it is rejected. if between then it is accepted
# if it is connected to a strong edge. (ratio should be 1:2
# or 1:3)
blur = cv2.Canny(blur, 50, 150)

cv2.imshow('result', blur)
cv2.waitKey(0)
