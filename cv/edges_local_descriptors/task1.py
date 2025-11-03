import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(image, ax, title):
	ax.hist(image.ravel(), bins=256, histtype='step', color='black')
	ax.set_title(title)
	ax.set_xlabel('Intensity')
	ax.set_ylabel('Frequency')
	ax.grid(True)

def calculate_stats(image):
	mean = np.mean(image)
	std = np.std(image)
	return mean, std

def rgb_to_gray_bt601(rgb_image):
	rgb_image = ensure_rgb(rgb_image)
	return 0.299 * rgb_image[..., 0] + 0.587 * rgb_image[..., 1] + 0.114 * rgb_image[..., 2]

def rgb_to_gray_avg(rgb_image):
	return np.mean(rgb_image, axis=2)

def ensure_rgb(image):
	if image.ndim == 2:
		return np.stack([image] * 3, axis=-1)
	elif image.ndim == 3:
		if image.shape[2] == 3:
			return image
		elif image.shape[2] == 4:
			return image[..., :3]
		else:
			raise ValueError("Изображение должно быть в формате RGB или оттенков серого")
	else:
		raise ValueError("Неизвестный формат изображения. Ожидается RGB или изображение в оттенках серого")

def showcase(image1, name1, image2, name2):
	image1_rgb = ensure_rgb(image1)
	image2_rgb = ensure_rgb(image2)

	# Преобразуем в оттенки серого
	gray_image1_bt601 = rgb_to_gray_bt601(image1_rgb)
	gray_image2_bt601 = rgb_to_gray_bt601(image2_rgb)
	gray_image1_avg = rgb_to_gray_avg(image1_rgb)
	gray_image2_avg = rgb_to_gray_avg(image2_rgb)

	# Создаем фигуру для отображения всех гистограмм и изображений
	fig, axes = plt.subplots(3, 4, figsize=(16, 12))

	# Отображение изображений и гистограмм на подграфиках
	axes[0, 0].imshow(gray_image1_bt601, cmap='gray')
	axes[0, 0].set_title(f'{name1} - BT.601')
	axes[0, 0].axis('off')

	axes[0, 1].imshow(gray_image1_avg, cmap='gray')
	axes[0, 1].set_title(f'{name1} - Average')
	axes[0, 1].axis('off')

	axes[0, 2].imshow(gray_image2_bt601, cmap='gray')
	axes[0, 2].set_title(f'{name2} - BT.601')
	axes[0, 2].axis('off')

	axes[0, 3].imshow(gray_image2_avg, cmap='gray')
	axes[0, 3].set_title(f'{name2} - Average')
	axes[0, 3].axis('off')

	# Гистограммы для изображений
	plot_histogram(gray_image1_bt601, axes[1, 0], f'{name1} - BT.601 Histogram')
	plot_histogram(gray_image1_avg, axes[1, 1], f'{name1} - Average Histogram')
	plot_histogram(gray_image2_bt601, axes[1, 2], f'{name2} - BT.601 Histogram')
	plot_histogram(gray_image2_avg, axes[1, 3], f'{name2} - Average Histogram')

	# Печать статистики для изображений
	mean_bt601_1, std_bt601_1 = calculate_stats(gray_image1_bt601)
	mean_avg_1, std_avg_1 = calculate_stats(gray_image1_avg)

	mean_bt601_2, std_bt601_2 = calculate_stats(gray_image2_bt601)
	mean_avg_2, std_avg_2 = calculate_stats(gray_image2_avg)

	axes[2, 0].text(0.1, 0.8, f"BT.601 ({name1})\nMean: {mean_bt601_1:.2f}, Std: {std_bt601_1:.2f}", fontsize=12)
	axes[2, 0].axis('off')

	axes[2, 1].text(0.1, 0.8, f"Average ({name1})\nMean: {mean_avg_1:.2f}, Std: {std_avg_1:.2f}", fontsize=12)
	axes[2, 1].axis('off')

	axes[2, 2].text(0.1, 0.8, f"BT.601 ({name2})\nMean: {mean_bt601_2:.2f}, Std: {std_bt601_2:.2f}", fontsize=12)
	axes[2, 2].axis('off')

	axes[2, 3].text(0.1, 0.8, f"Average ({name2})\nMean: {mean_avg_2:.2f}, Std: {std_avg_2:.2f}", fontsize=12)
	axes[2, 3].axis('off')

	plt.tight_layout()
	plt.show()
