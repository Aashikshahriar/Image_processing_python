import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.restoration import (denoise_wavelet,estimate_sigma)
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio

#reading images
img=cv.imread('cloud.jpeg',0)
img=np.float64(img)

sigma=0.01
nimg=random_noise(img,var=sigma**2)

sigma_est=estimate_sigma(nimg,average_sigmas=True)

#denoising using bayes
img_bayes=denoise_wavelet(nimg,method='BayesShrink',mode='soft',wavelet_levels=3,wavelet='bior6.8',rescale_sigma=True)

#denoising using VisuShrink
img_visu=denoise_wavelet(nimg,method='VisuShrink',mode='soft',sigma=sigma_est/3,wavelet_levels=5,wavelet='bior6.8',rescale_sigma=True)

#finding PSNR


#plotting images
plt.figure(figsize=(30,30))
plt.subplot(2,2,1)
plt.imshow(img,cmap=plt.cm.gray)
plt.title('Original image')

plt.subplot(2,2,2)
plt.imshow(nimg,cmap=plt.cm.gray)
plt.title('Noisy image')

plt.subplot(2,2,3)
plt.imshow(img_bayes,cmap=plt.cm.gray)
plt.title('Denoised image using bayes')

plt.subplot(2,2,4)
plt.imshow(img_visu,cmap=plt.cm.gray)
plt.title('Denoised image using Visushrink')

plt.show()

#printing PSNR
print('PSNR [original vs noisy image]:',psnr_noisy)
print('PSNR [original vs Denoised(Bayes)]:',psnr_bayes)
print('PSNR [original vs Denoised(Visu)]:',psnr_visu)