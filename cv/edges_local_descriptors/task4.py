import numpy as np
import cv2
import matplotlib.pyplot as plt
from task1 import rgb_to_gray_bt601

def compute_gradients(image):
    image = image.astype(np.float32)
    Ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(Ix**2 + Iy**2)
    orientation = np.rad2deg(np.arctan2(Iy, Ix)) % 180
    return magnitude, orientation

def cell_histogram(mag, ang, cell_size=8, bins=9):
    bin_width = 180 // bins
    hist = np.zeros(bins, dtype=np.float32)

    for i in range(cell_size):
        for j in range(cell_size):
            m = mag[i, j]
            a = ang[i, j]
            
            bin_idx = int(a // bin_width)
            next_bin = (bin_idx + 1) % bins
            ratio = (a - bin_idx*bin_width) / bin_width
            
            hist[bin_idx] += m * (1 - ratio)
            hist[next_bin] += m * ratio
    return hist

def compute_cell_histograms(image, cell_size=8, bins=9):
    h, w = image.shape
    mag, ang = compute_gradients(image)

    n_cells_x = w // cell_size
    n_cells_y = h // cell_size

    cell_hists = np.zeros((n_cells_y, n_cells_x, bins), dtype=np.float32)
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_mag = mag[i*cell_size:(i+1)*cell_size,
                           j*cell_size:(j+1)*cell_size]
            cell_ang = ang[i*cell_size:(i+1)*cell_size,
                           j*cell_size:(j+1)*cell_size]
            cell_hists[i, j, :] = cell_histogram(cell_mag, cell_ang, cell_size, bins)
    return cell_hists

def visualize_hog(image, title, resize=(128, 128), cell_size=8, bins=9):
    image = image / 255.0
    
    orig_h, orig_w = image.shape[:2]
    if resize is not None:
        small_img = cv2.resize(image, resize)
    else:
        small_img = image.copy()

    gray_small = rgb_to_gray_bt601(small_img).astype(np.float32) / 255.0

    cell_hists = compute_cell_histograms(gray_small, cell_size, bins)

    n_cells_y, n_cells_x, _ = cell_hists.shape
    bin_width = 180 // bins

    scale_x = orig_w / small_img.shape[1]
    scale_y = orig_h / small_img.shape[0]

    plt.figure(figsize=(8,8))
    plt.imshow(image)

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            hist = cell_hists[i, j, :]
            hist /= hist.max() + 1e-6  # нормировка для визуализации

            y0_small = i * cell_size + cell_size // 2
            x0_small = j * cell_size + cell_size // 2

            y0 = y0_small * scale_y
            x0 = x0_small * scale_x

            for b in range(bins):
                theta = (b + 0.5) * bin_width * np.pi / 180
                dx = np.cos(theta) * hist[b] * (cell_size // 2) * scale_x
                dy = np.sin(theta) * hist[b] * (cell_size // 2) * scale_y
                plt.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy],
                         color='red', linewidth=1, alpha=0.7)

    plt.title(f"{title} (cell_size={cell_size}x{cell_size} bins={bins})")
    plt.axis('off')
    plt.show()
