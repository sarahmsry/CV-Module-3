import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_gaussian_kernel(size, sigma):
	"""Create 2D Gaussian kernel for blurring"""
	ax = np.arange(-size // 2 + 1., size // 2 + 1.)
	xx, yy = np.meshgrid(ax, ax)
	kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
	return kernel / np.sum(kernel)

def spatial_filtering(img, kernel):
	"""convolution in spatial domain"""
	return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_WRAP)

def frequency_filtering(img, kernel):
	"""multiplication in frequency domain"""
	H, W = img.shape
	kh, kw = kernel.shape
	
	# Get optimal size
	dft_H = cv2.getOptimalDFTSize(H + kh - 1)
	dft_W = cv2.getOptimalDFTSize(W + kw - 1)
	
	# Transform image to frequency domain
	img_padded = np.zeros((dft_H, dft_W), dtype=np.float32)
	img_padded[:H, :W] = img
	img_dft = cv2.dft(img_padded, flags=cv2.DFT_COMPLEX_OUTPUT)
	
	# Transform kernel to frequency domain
	kernel_padded = np.zeros((dft_H, dft_W), dtype=np.float32)
	kernel_padded[:kh, :kw] = kernel
	kernel_dft = cv2.dft(kernel_padded, flags=cv2.DFT_COMPLEX_OUTPUT)
	
	# Multiply in frequency domain 
	result_dft = cv2.mulSpectrums(img_dft, kernel_dft, 0)
	
	# Transform back to spatial domain
	result = cv2.idft(result_dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
	
	# Extract same size as input
	start_i, start_j = kh // 2, kw // 2
	return result[start_i:start_i + H, start_j:start_j + W]

def main():
	img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
	img = img.astype(np.float32) / 255.0
	
	# Create Gaussian blur kernel
	kernel = create_gaussian_kernel(size=21, sigma=5.0)
	
	print("Convolution in Space = Multiplication in Frequency")
		
	# First method: Spatial filtering (convolution)
	print("\n Spatial Filtering Convolution)")
	blurred_spatial = spatial_filtering(img, kernel)
	
	# Second method: Frequency domain filtering (multiplication)
	print("Frequency Filtering (Multiplication in Fourier domain)")
	blurred_freq = frequency_filtering(img, kernel)
	
	# Compare results
	diff = blurred_spatial - blurred_freq
	max_diff = np.max(np.abs(diff))
	mean_diff = np.mean(np.abs(diff))
	
	print("PROOF OF EQUIVALENCE:")
	print(f"Maximum difference:      {max_diff:.6e}")
	print(f"Mean absolute difference: {mean_diff:.6e}")
	print(f"Correlation coefficient:  {np.corrcoef(blurred_spatial.flatten(), blurred_freq.flatten())[0,1]:.15f}")
	
	fig, axes = plt.subplots(1, 3, figsize=(16, 4))
	
	axes[0].imshow(img, cmap='gray')
	axes[0].set_title('Original Image')
	axes[0].axis('off')
	
	axes[1].imshow(blurred_spatial, cmap='gray')
	axes[1].set_title('Spatial Domain\n(Convolution)')
	axes[1].axis('off')
	
	axes[2].imshow(blurred_freq, cmap='gray')
	axes[2].set_title('Frequency Domain\n(Multiplication)')
	axes[2].axis('off')
	
	fig.suptitle('Proof: Convolution in Space = Multiplication in Frequency', 
	             fontsize=14, fontweight='bold')
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()