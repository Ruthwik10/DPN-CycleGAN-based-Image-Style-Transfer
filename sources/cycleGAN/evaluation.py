import os
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(img1, img2):   ## Rutwhik Adapala
    """Calculates the PSNR between two images."""
    img1 = np.array(img1)
    img2 = np.array(img2)
    return peak_signal_noise_ratio(img1, img2, data_range=img1.max() - img1.min())

from skimage.metrics import structural_similarity

def calculate_ssim(img1, img2):  ## Ruthwik Adapala
    """Calculates the SSIM between two images with a custom window size."""
    img1 = np.array(img1)
    img2 = np.array(img2)

    
    min_dim = min(img1.shape[0], img1.shape[1])  # Get the smallest dimension (height or width)
    if min_dim < 7:
        raise ValueError("Image dimensions must be at least 7x7 pixels.")

    win_size = min(7, min_dim)
    if win_size % 2 == 0:  # Ensure `win_size` is odd
        win_size -= 1
    
    # Calculate SSIM with the appropriate window size and `channel_axis` for color images
    return structural_similarity(img1, img2, win_size=win_size, channel_axis=-1)


def evaluate_metrics_from_single_dir(directory, domain='a'):
    """Evaluates PSNR and SSIM metrics between real and generated images in a single directory."""
    psnr_scores = []
    ssim_scores = []

    # Collect real and generated image names for the specified domain
    real_images = sorted([f for f in os.listdir(directory) if f.startswith(f"{domain}_real")])
    generated_images = sorted([f for f in os.listdir(directory) if f.startswith(f"{domain}_gen")])

    # Ensure we have matching real and generated images
    for real_img_name, gen_img_name in zip(real_images, generated_images):
        real_img_path = os.path.join(directory, real_img_name)
        gen_img_path = os.path.join(directory, gen_img_name)

        if os.path.isfile(real_img_path) and os.path.isfile(gen_img_path):
            real_img = Image.open(real_img_path).convert('RGB')
            gen_img = Image.open(gen_img_path).convert('RGB')

            psnr = calculate_psnr(real_img, gen_img)
            ssim = calculate_ssim(real_img, gen_img)

            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    print(f"Average PSNR for Domain {domain.upper()}: {avg_psnr:.2f}")
    print(f"Average SSIM for Domain {domain.upper()}: {avg_ssim:.4f}")

    return avg_psnr, avg_ssim

# Example usage
directory_path = 'results'

print("Metrics for Domain A:")
evaluate_metrics_from_single_dir(directory_path, domain='a')

print("\nMetrics for Domain B:")
evaluate_metrics_from_single_dir(directory_path, domain='b')
