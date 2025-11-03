import numpy as np
import cv2
import matplotlib.pyplot as plt
from task1 import rgb_to_gray_bt601
from skimage import data
from task2 import sobel_gradients, magnitude, angle

def gaussian_blur(image, sigma):
    blurred_image = cv2.GaussianBlur(image, (5, 5), sigma)
    return blurred_image

def non_maximum_suppression(magnitude, angle):
    Z = np.zeros_like(magnitude)
    angle = angle / 180.0 * np.pi
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            quantized_angle = quantize_angle(angle[i, j])
            q = 255
            r = 255
            if quantized_angle == 0:
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif quantized_angle == 45:
                q = magnitude[i + 1, j + 1]
                r = magnitude[i - 1, j - 1]
            elif quantized_angle == 90:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif quantized_angle == 135:
                q = magnitude[i - 1, j + 1]
                r = magnitude[i + 1, j - 1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0
    return Z

def quantize_angle(angle):
    angle = angle % 180
    
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        return 0
    elif (22.5 <= angle < 67.5):
        return 45
    elif (67.5 <= angle < 112.5):
        return 90
    elif (112.5 <= angle < 157.5):
        return 135

def double_thresholding(magnitude, low_threshold, high_threshold):
    strong_edges = (magnitude > high_threshold)
    weak_edges = ((magnitude >= low_threshold) & (magnitude <= high_threshold))
    return strong_edges, weak_edges

def hysteresis(strong_edges, weak_edges):
    edges = np.copy(strong_edges)
    
    for i in range(1, strong_edges.shape[0] - 1):
        for j in range(1, strong_edges.shape[1] - 1):
            if weak_edges[i, j]:
                if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                    edges[i, j] = True
    return edges

def compare_with_opencv_canny(image):
    image = np.clip(image, 0, 255).astype(np.uint8)

    t_low = 200
    coef = 2
    edges_canny = cv2.Canny(image, t_low, t_low * coef)
    return edges_canny

def plot_gaussian(image, sigma_values):
    gray_image = rgb_to_gray_bt601(image)
    plt.figure(figsize=(15, 5))
    n = len(sigma_values)
    
    for i, sigma in enumerate(sigma_values):
        blurred = gaussian_blur(gray_image, sigma)
        
        plt.subplot(1, n, i + 1)
        plt.imshow(blurred, cmap='gray')
        plt.title(f"Gaussian σ={sigma}")
        plt.axis('off')
    
    plt.suptitle("Сравнение Gaussian Blur при разных σ", fontsize=14)
    plt.show()

def canny_edge_detection(image, sigma_values, low_threshold, high_threshold):
    gray_image = rgb_to_gray_bt601(image)

    for sigma in sigma_values:
        blurred_image = gaussian_blur(gray_image, sigma)

        Ix, Iy = sobel_gradients(blurred_image)
        _magnitude = magnitude(Ix, Iy)
        _angle = angle(Ix, Iy)

        nms_image = non_maximum_suppression(_magnitude, _angle)

        strong_edges, weak_edges = double_thresholding(nms_image, low_threshold, high_threshold)

        final_edges = hysteresis(strong_edges, weak_edges)

        edges_canny = compare_with_opencv_canny(gray_image)

        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(blurred_image, cmap='gray')
        plt.title(f"Gaussian Blur (sigma={sigma})")
        
        plt.subplot(2, 3, 2)
        plt.imshow(_magnitude, cmap='gray')
        plt.title("Magnitude")
        
        plt.subplot(2, 3, 3)
        plt.imshow(nms_image, cmap='gray')
        plt.title("NMS")
        
        plt.subplot(2, 3, 4)
        plt.imshow(final_edges, cmap='gray')
        plt.title("Final Edges (after Hysteresis)")
        
        plt.subplot(2, 3, 5)
        plt.imshow(edges_canny, cmap='gray')
        plt.title("OpenCV Canny")
        
        plt.show()
