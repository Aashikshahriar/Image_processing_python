import cv2
import numpy as np
import matplotlib.pyplot as plt

def log_edge_detection(image_path, sigma=1):
  """
  Performs Laplacian of Gaussian edge detection on an image.

  Args:
    image_path: Path to the image file.
    sigma: Standard deviation for the Gaussian kernel.

  Returns:
    The LoG edge detected image.
  """

  try:
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert to float32 for better precision
    img = np.float64(img)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (3, 3), sigma)

    # Compute Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Convert back to uint8 for display
    laplacian = np.uint8(np.absolute(laplacian))

    return laplacian

  except Exception as e:
    print(f"Error: {e}")
    return None

# Example usage
image_path = 'ashik.jpg'
edge_image = log_edge_detection(image_path)

if edge_image is not None:
  cv2.imshow('LoG Edge', edge_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  plt.imshow(edge_image)
  plt.waitforbuttonpress()
