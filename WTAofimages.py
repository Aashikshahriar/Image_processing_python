import numpy as np
import cv2 as cv
import pywt
import pywt.data
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img=cv.imread('example.bmp',0)
coeffs2=pywt.dwt2(img,'db3',mode='periodization')
cA,(cH,cV,cD)=coeffs2
imgr=pywt.idwt2(coeffs2,'db3',mode='periodization')
imgr=np.uint8(imgr)
plt.figure(figsize=(30,30))
plt.subplot(2,2,1)
plt.imshow(cA)
plt.title('cA:Aproximation coeff',fontsize=20)
plt.subplot(2,2,2)
plt.imshow(cH)
plt.title('cH:Horizontal coeff',fontsize=20)
plt.subplot(2,2,3)
plt.imshow(cV)
plt.title('cV:Vertical coeff',fontsize=20)
plt.subplot(2,2,4)
plt.imshow(cD)
plt.title('cD:Diagonal coeff',fontsize=20)
plt.show()
