import cv2
import numpy as np
import matplotlib.pyplot as plt

def bit_plane_slicing(img, plane):
 
  # Create a mask for the desired bit plane
  mask = np.uint8(1 << plane)

  # Extract the bit plane using bitwise AND
  bit_plane = np.bitwise_and(img, mask)

  # Normalize the bit plane for visualization
  bit_plane = bit_plane * 255

  return bit_plane

# Load the image
img = cv2.imread('heart.jpeg', 0)  # Load as grayscale

# Extract bit planes
for i in range(8):
  bit_plane = bit_plane_slicing(img, i)
  #cv2.imshow(f"Bit Plane {i}", bit_plane)
  if i==7:
      a,b=bit_plane.shape
      bsimage=bit_plane
  plt.subplot(3,3,i+1)
  plt.imshow(bit_plane)
  plt.title('For bit level'+str(i))

print(a,b) 
print(img.shape)
cv2.imwrite('heartbls.jpeg',bsimage)

plt.waitforbuttonpress()
