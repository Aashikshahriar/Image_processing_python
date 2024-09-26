import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img_path='corner.png'
img=cv.imread(img_path)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray=np.float32(gray)
dst=cv.cornerHarris(gray,2,3,0.07)
dst=cv.dilate(dst,None)
img[dst > 0.01 * dst.max()]=[255, 0, 0] 
plt.imshow(img,cmap='gray')
plt.waitforbuttonpress()