import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_pepper_noise(image, amount=0.05):
    """Add salt and pepper noise to image"""
    noisy = np.copy(image)
    
    # Salt noise (white pixels)
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 255
    
    # Pepper noise (black pixels)
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0
    
    return noisy

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to image"""
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(image + gaussian, 0, 255).astype(np.uint8)
    return noisy

def add_uniform_noise(image, low=-50, high=50):
    """Add uniform noise to image"""
    uniform = np.random.uniform(low, high, image.shape)
    noisy = np.clip(image + uniform, 0, 255).astype(np.uint8)
    return noisy

def apply_mean_filter(image, kernel_size=5):
    """Apply mean filter for denoising"""
    return cv2.blur(image, (kernel_size, kernel_size))

def apply_median_filter(image, kernel_size=5):
    """Apply median filter for denoising"""
    return cv2.medianBlur(image, kernel_size)

def apply_gaussian_filter(image, kernel_size=5):
    """Apply Gaussian filter for denoising"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_bilateral_filter(image):
    """Apply bilateral filter for denoising"""
    return cv2.bilateralFilter(image, 9, 75, 75)

def display_images(images, titles, rows=2, cols=3, figsize=(15, 10)):
    """Helper function to display multiple images"""
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        if len(img.shape) == 2:  # Grayscale
            plt.imshow(img, cmap='gray')
        else:  # Color
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title, fontsize=12, fontweight='bold')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def calculate_psnr(original, denoised):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def main():
    # Load image
    image_path = 'Test Flower.jpg'  # Change to your image path
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    
    print("Noise Addition and Removal")
    print("=" * 60)
    
    # ========== 1. SALT & PEPPER NOISE ==========
    print("\n1. Adding Salt & Pepper Noise...")
    sp_noisy = add_salt_pepper_noise(image, amount=0.05)
    
    images = [image, sp_noisy]
    titles = ['Original Image', 'Salt & Pepper Noise (5%)']
    display_images(images, titles, rows=1, cols=2, figsize=(12, 5))
    
    # ========== 2. DENOISING SALT & PEPPER NOISE ==========
    print("\n2. Denoising Salt & Pepper Noise with different filters...")
    
    sp_mean = apply_mean_filter(sp_noisy, kernel_size=5)
    sp_median = apply_median_filter(sp_noisy, kernel_size=5)
    sp_gaussian = apply_gaussian_filter(sp_noisy, kernel_size=5)
    sp_bilateral = apply_bilateral_filter(sp_noisy)
    
    images = [image, sp_noisy, sp_mean, sp_median, sp_gaussian, sp_bilateral]
    titles = [
        'Original', 'Salt & Pepper Noisy',
        'Mean Filter (5x5)', 'Median Filter (5x5)',
        'Gaussian Filter (5x5)', 'Bilateral Filter'
    ]
    display_images(images, titles, rows=2, cols=3)
    
    # Calculate PSNR for Salt & Pepper
    print("\nPSNR for Salt & Pepper Noise Removal:")
    print(f"Mean Filter: {calculate_psnr(image, sp_mean):.2f} dB")
    print(f"Median Filter: {calculate_psnr(image, sp_median):.2f} dB")
    print(f"Gaussian Filter: {calculate_psnr(image, sp_gaussian):.2f} dB")
    print(f"Bilateral Filter: {calculate_psnr(image, sp_bilateral):.2f} dB")
    
    # ========== 3. GAUSSIAN NOISE ==========
    print("\n3. Adding Gaussian Noise...")
    gaussian_noisy = add_gaussian_noise(image, mean=0, sigma=25)
    
    images = [image, gaussian_noisy]
    titles = ['Original Image', 'Gaussian Noise (σ=25)']
    display_images(images, titles, rows=1, cols=2, figsize=(12, 5))
    
    # ========== 4. DENOISING GAUSSIAN NOISE ==========
    print("\n4. Denoising Gaussian Noise with different filters...")
    
    g_mean = apply_mean_filter(gaussian_noisy, kernel_size=5)
    g_median = apply_median_filter(gaussian_noisy, kernel_size=5)
    g_gaussian = apply_gaussian_filter(gaussian_noisy, kernel_size=5)
    g_bilateral = apply_bilateral_filter(gaussian_noisy)
    
    images = [image, gaussian_noisy, g_mean, g_median, g_gaussian, g_bilateral]
    titles = [
        'Original', 'Gaussian Noisy',
        'Mean Filter (5x5)', 'Median Filter (5x5)',
        'Gaussian Filter (5x5)', 'Bilateral Filter'
    ]
    display_images(images, titles, rows=2, cols=3)
    
    # Calculate PSNR for Gaussian
    print("\nPSNR for Gaussian Noise Removal:")
    print(f"Mean Filter: {calculate_psnr(image, g_mean):.2f} dB")
    print(f"Median Filter: {calculate_psnr(image, g_median):.2f} dB")
    print(f"Gaussian Filter: {calculate_psnr(image, g_gaussian):.2f} dB")
    print(f"Bilateral Filter: {calculate_psnr(image, g_bilateral):.2f} dB")
    
    # ========== 5. UNIFORM NOISE ==========
    print("\n5. Adding Uniform Noise...")
    uniform_noisy = add_uniform_noise(image, low=-50, high=50)
    
    images = [image, uniform_noisy]
    titles = ['Original Image', 'Uniform Noise']
    display_images(images, titles, rows=1, cols=2, figsize=(12, 5))
    
    # ========== 6. COMPARISON OF KERNEL SIZES (MEDIAN FILTER) ==========
    print("\n6. Comparing different kernel sizes (Median Filter on Salt & Pepper)...")
    
    median_3 = apply_median_filter(sp_noisy, kernel_size=3)
    median_5 = apply_median_filter(sp_noisy, kernel_size=5)
    median_7 = apply_median_filter(sp_noisy, kernel_size=7)
    median_9 = apply_median_filter(sp_noisy, kernel_size=9)
    
    images = [sp_noisy, median_3, median_5, median_7, median_9]
    titles = ['Noisy', 'Median 3x3', 'Median 5x5', 'Median 7x7', 'Median 9x9']
    display_images(images, titles, rows=2, cols=3)
    
    # ========== 7. COMPARISON OF KERNEL SIZES (MEAN FILTER) ==========
    print("\n7. Comparing different kernel sizes (Mean Filter on Gaussian)...")
    
    mean_3 = apply_mean_filter(gaussian_noisy, kernel_size=3)
    mean_5 = apply_mean_filter(gaussian_noisy, kernel_size=5)
    mean_7 = apply_mean_filter(gaussian_noisy, kernel_size=7)
    mean_9 = apply_mean_filter(gaussian_noisy, kernel_size=9)
    
    images = [gaussian_noisy, mean_3, mean_5, mean_7, mean_9]
    titles = ['Noisy', 'Mean 3x3', 'Mean 5x5', 'Mean 7x7', 'Mean 9x9']
    display_images(images, titles, rows=2, cols=3)
    
    # ========== 8. SAVE RESULTS ==========
    print("\n8. Saving results...")
    cv2.imwrite('salt_pepper_noisy.jpg', sp_noisy)
    cv2.imwrite('gaussian_noisy.jpg', gaussian_noisy)
    cv2.imwrite('uniform_noisy.jpg', uniform_noisy)
    cv2.imwrite('sp_median_denoised.jpg', sp_median)
    cv2.imwrite('gaussian_mean_denoised.jpg', g_mean)
    cv2.imwrite('bilateral_denoised.jpg', sp_bilateral)
    
    print("\nAll operations completed successfully!")
    print("Results saved to current directory.")
    print("\nSummary:")
    print("- Best for Salt & Pepper: Median Filter")
    print("- Best for Gaussian: Gaussian/Mean Filter")
    print("- Bilateral Filter: Preserves edges while denoising")

if __name__ == "__main__":
    main()
