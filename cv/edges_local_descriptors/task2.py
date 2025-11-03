import numpy as np
import matplotlib.pyplot as plt
from task1 import rgb_to_gray_bt601
import cv2

def convolve2d(image, kernel, padding_type='reflect'):
    if len(image.shape) == 3:
        image = rgb_to_gray_bt601(image)
    image = image.astype(np.float64)

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    if padding_type == 'zero':
        padded = np.pad(image, ((pad_h, pad_h),(pad_w,pad_w)), mode='zero', constant_values=0)
    elif padding_type == 'reflect':
        padded = np.pad(image, ((pad_h, pad_h),(pad_w,pad_w)), mode='reflect')
    else:
        raise ValueError("padding_type должен иметь значение 'constant'/'zero' или 'reflect'")

    out = np.zeros_like(image, dtype=np.float64)

    for i in range(image_height):
        for j in range(image_width):
            region = padded[i:i+kernel_height, j:j+kernel_width]
            out[i, j] = np.sum(region * kernel)
    return out

def sobel_gradients(image, padding='reflect'):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float64)
    Ix = convolve2d(image, sobel_x, padding_type=padding)
    Iy = convolve2d(image, sobel_y, padding_type=padding)
    return Ix, Iy

def magnitude(Ix, Iy):
    return np.sqrt(Ix**2 + Iy**2)

def angle(Ix, Iy):
    return np.arctan2(Iy, Ix)

def compare_with_opencv_sobel(image, padding='reflect'):
    img = image.astype(np.float64)
    border = cv2.BORDER_REFLECT if padding == 'reflect' else cv2.BORDER_CONSTANT
    Ix_cv = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=border)
    Iy_cv = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, borderType=border)
    M_cv = np.sqrt(Ix_cv**2 + Iy_cv**2)
    Theta_cv = np.arctan2(Iy_cv, Ix_cv)
    return Ix_cv, Iy_cv, M_cv, Theta_cv

def psnr(image1, image2, data_range=None):
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    mse_val = np.mean((image1 - image2)**2)
    if mse_val == 0:
        return float('inf')
    if data_range is None:
        PIXEL_MAX = max(image1.max(), image2.max(), 1.0)
    else:
        PIXEL_MAX = data_range
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse_val))

def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def run_sobel(image):
    image = rgb_to_gray_bt601(image)
    
    Ix, Iy = sobel_gradients(image)
    Ix = np.abs(Ix)
    Iy = np.abs(Iy)
    
    M = magnitude(Ix, Iy)
    Theta = angle(Ix, Iy)
    
    Ix_cv, Iy_cv, M_opencv, Theta_opencv = compare_with_opencv_sobel(image)
    
    mse_M = mse(M, M_opencv)
    psnr_M = psnr(M, M_opencv)
    mse_Theta = mse(Theta, Theta_opencv)
    psnr_Theta = psnr(Theta, Theta_opencv)

    print(f"MSE (Magnitude): {mse_M:.4f}")
    print(f"PSNR (Magnitude): {psnr_M:.4f} dB")
    print(f"MSE (Theta): {mse_Theta:.4f}")
    print(f"PSNR (Theta): {psnr_Theta:.4f} dB")

    # Выводим результаты
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(Ix, cmap='gray')
    plt.title("Ix (X-gradient)")
    
    plt.subplot(2, 3, 2)
    plt.imshow(Iy, cmap='gray')
    plt.title("Iy (Y-gradient)")
    
    plt.subplot(2, 3, 3)
    plt.imshow(M, cmap='gray')
    plt.title("Magnitude (M)")
    
    plt.subplot(2, 3, 4)
    plt.imshow(Theta, cmap='gray')
    plt.title("Theta (Angle)")
    
    plt.subplot(2, 3, 5)
    plt.imshow(M_opencv, cmap='gray')
    plt.title("M (OpenCV Sobel)")
    
    plt.subplot(2, 3, 6)
    plt.imshow(Theta_opencv, cmap='gray')
    plt.title("Theta (OpenCV Sobel)")
    
    plt.show()

def plot_padding_artefacts(image):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    gray_image = rgb_to_gray_bt601(image)

    padded_image_zero = np.pad(gray_image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    padded_image_reflect = np.pad(gray_image, ((1, 1), (1, 1)), mode='reflect')
    
    axes[0].imshow(padded_image_zero, cmap='gray')
    axes[0].set_title(f"Padded Image (Zero Padding)")
    axes[0].axis('off')
    
    axes[1].imshow(padded_image_reflect, cmap='gray')
    axes[1].set_title(f"Padded Image (Reflect Padding)")
    axes[1].axis('off')
    
    plt.show()
