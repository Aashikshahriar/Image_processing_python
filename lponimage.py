import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

image=cv.imread('seyam.png')
kernel=7
sigma=5
blurred_img=cv.GaussianBlur(image,(kernel,kernel),sigma) # here kernel is kernel size(usually odd) and 1 is value of sigma
cv.imwrite('blurred_seyam.png',blurred_img)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('original image')
plt.imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Gaussian Blurred Image')
plt.imshow(cv.cvtColor(blurred_img,cv.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
