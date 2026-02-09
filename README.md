# Image Blurring: Spatial vs Frequency Domain
## CV-Module-3 | Computer Vision Spring 2026 Week 3

### Overview
This project demonstrates that **convolution in the spatial domain is equivalent to multiplication in the frequency (Fourier) domain**.

By implementing image blurring (Gaussian filter) using two different approaches, this module provides proof of the convolution-multiplication duality. 

### Key Concepts

#### Spatial Domain Filtering (Convolution)
- Apply a filter kernel directly to the image by sliding it across every pixel
- Each output pixel is computed as a weighted sum of neighboring pixels
- **Pros**: Intuitive, works directly on pixel values
- **Cons**: Can be computationally expensive for large kernels

#### Frequency Domain Filtering (Multiplication)
- Transform the image and kernel to the frequency domain using FFT (Fast Fourier Transform)
- Multiply the frequency representations
- Transform the result back to the spatial domain using inverse FFT

### Implementation Details

**Gaussian Kernel**: A 2D Gaussian function is used as the blur filter

**Methods**:
1. `spatial_filtering()`: Uses OpenCV's `cv2.filter2D()` for direct convolution
2. `frequency_filtering()`: Uses OpenCV's DFT functions to multiply in frequency domain

### Usage

#### Requirements
```
opencv-python
numpy
matplotlib
```

#### Installation
```bash
pip install opencv-python numpy matplotlib
```

#### Running the Demo
```bash
python image_blurring.py
```

This will:
1. Load a test image
2. Create a 21×21 Gaussian blur kernel (σ=5)
3. Apply blurring using both spatial and frequency methods
4. Display the original, spatial-blurred, and frequency-blurred images
5. Print numerical differences between the two methods

#### Expected Output
The console output shows the equivalence:
```
Convolution in Space = Multiplication in Frequency

Spatial Filtering (Convolution)
Frequency Filtering (Multiplication in Fourier domain)

PROOF OF EQUIVALENCE:
Maximum difference:      < 1e-10
Mean absolute difference: < 1e-12
Correlation coefficient:  0.999999999999999
```

The tiny differences are due to floating-point arithmetic precision, confirming the mathematical equivalence.

### Results

The generated `image_blur_comparison.png` shows:
- **Left**: Original image
- **Center**: Result from spatial domain convolution
- **Right**: Result from frequency domain multiplication

The three images appear identical visually, while the console prints the numerical differences, proving they are mathematically equivalent (within machine precision).
