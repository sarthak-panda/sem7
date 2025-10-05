import numpy as np
import cv2
import matplotlib.pyplot as plt


def resize_naive(image, k):
    h, w = image.shape
    new_h, new_w = int(h * k), int(w * k)
    
    # Use bicubic interpolation as a high-quality "naive" method
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized_image

def resize_filtered(image, k):
  
    h, w = image.shape
    
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    mask = np.zeros_like(image, dtype=float)
    center_h, center_w = h // 2, w // 2
    cutoff_h = int(h * k / 2)
    cutoff_w = int(w * k / 2)
    
    mask[center_h - cutoff_h : center_h + cutoff_h, center_w - cutoff_w : center_w + cutoff_w] = 1.0


    f_transform_filtered = f_transform_shifted * mask
    
    
    f_inverse_shifted = np.fft.ifftshift(f_transform_filtered)
    image_filtered = np.fft.ifft2(f_inverse_shifted)
    image_filtered = np.real(image_filtered)

   
    step = int(1 / k)
    resized_image = image_filtered[::step, ::step]
    
    return resized_image

def main():
   
    
    image_path = "../Testcases/barbara.bmp"
    
    # Load the image from the file in grayscale
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    # Convert image to float and normalize to the [0, 1] range for processing
    original_image = original_image.astype(np.float32) / 255.0


    original_h, original_w = original_image.shape
    k_values = [1/2,1/4,1/8]
    
  
    num_k = len(k_values)
    plt.figure(figsize=(15, 5 * num_k))
    plt.gray()

   
    for i, k in enumerate(k_values):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        plt.gray()

        # Naive Resizing
        shrunk_naive = resize_naive(original_image, k)
        zoomed_naive = cv2.resize(shrunk_naive, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        axs[0].imshow(zoomed_naive)
        axs[0].set_title(f'Naive Resizing (k={k})')
        axs[0].axis('off')

        # Filtered Resizing
        shrunk_filtered = resize_filtered(original_image, k)
        zoomed_filtered = cv2.resize(shrunk_filtered, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        axs[1].imshow(zoomed_filtered)
        axs[1].set_title(f'Filtered Resizing (k={k})')
        axs[1].axis('off')

        # Original Image
        axs[2].imshow(original_image)
        axs[2].set_title('Original Image')
        axs[2].axis('off')

        fig.suptitle(f"Image Resizing Comparison (k={k})", fontsize=16)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
