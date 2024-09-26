import cv2
import numpy as np
from matplotlib import pyplot as plt

def plot_histogram(image, title, ax):
    # Compute histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # Plot histogram
    ax.plot(hist, color='black')
    ax.set_xlim([0, 256])
    ax.set_title(title)

def histogram_equalization(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image was loaded successfully
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Perform histogram equalization
    equalized_image = cv2.equalizeHist(image)

    # Save the equalized image
    cv2.imwrite(output_path, equalized_image)

    # Plot original and equalized images along with their histograms
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot original image and its histogram
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    plot_histogram(image, 'Original Histogram', axs[1, 0])

    # Plot equalized image and its histogram
    axs[0, 1].imshow(equalized_image, cmap='gray')
    axs[0, 1].set_title('Equalized Image')
    axs[0, 1].axis('off')

    plot_histogram(equalized_image, 'Equalized Histogram', axs[1, 1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_image_path = "seyam.png"  # Replace with your image path
    output_image_path = "histeq_seyam.png"  # Replace with the path to save the equalized image
    histogram_equalization(input_image_path, output_image_path)
