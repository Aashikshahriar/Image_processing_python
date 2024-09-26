import cv2
import numpy as np

def sobel_edge_detection(image_path):

  # Load the image
  img = cv2.imread(image_path, 0)  # Load as grayscale

  # Sobel x and y gradients
  sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
  sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

  # Calculate the magnitude of the gradient
  magnitude = np.sqrt(sobelx**2 + sobely**2)

  # Convert to 8-bit unsigned integer
  magnitude = np.uint8(magnitude)

  return magnitude

# Example usage
image_path = 'seyam.png'
edge_image = sobel_edge_detection(image_path)

# Display the edge image
cv2.imshow('Sobel Edge', edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
