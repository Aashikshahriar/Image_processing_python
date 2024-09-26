import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image_path='ashik.jpg'
threshold1=100
threshold2=200
img=cv.imread(image_path,0)
can_edge=cv.Canny(img,threshold1,threshold2)
plt.imshow(can_edge,cmap='gray')
plt.title('Canny Edge Detection')
plt.waitforbuttonpress()
