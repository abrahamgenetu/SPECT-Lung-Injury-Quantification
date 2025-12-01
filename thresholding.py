import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def read_and_process_image(filepath, image_size):
    """Read a binary image file and reshape it."""
    print(f"Reading and processing image: {filepath}")
    with open(filepath, 'rb') as f:
        img = np.fromfile(f, dtype=np.float32).reshape(image_size)
        img = np.rot90(img, k=3)  # Rotate the image
    return img

def average_images(file_list, image_size):
    """Calculate the average of a list of images."""
    print(f"Calculating average for {len(file_list)} images.")
    return np.mean([read_and_process_image(f, image_size) for f in file_list], axis=0)

def create_and_save_thresholded_images(maa_image, average_du, average_hmpao, percentages, output_dir, image_index):
    """Create and save thresholded images based on intensity percentages."""
    max_intensity_value = np.max(maa_image)
    maa_diff = maa_image - average_du - average_hmpao
    
    for i, percentage in enumerate(percentages):
        threshold_value = max_intensity_value * percentage
        print(f"Applying threshold: {threshold_value}")
        roi_mask = (maa_diff >= threshold_value).astype(np.uint8) * 255
        
        # Apply Gaussian filter for smoothing
        roi_mask_smoothed = gaussian_filter(roi_mask, sigma=1)
        roi_mask_smoothed = (roi_mask_smoothed > 128).astype(np.uint8) * 255
        
        # Save each thresholded image
        mask_image = Image.fromarray(roi_mask_smoothed)
        mask_image.save(os.path.join(output_dir, f'thresholded_{image_index + 21:03d}_{i}.png'))

def create_lung_roi_dataset(base_path, output_dir, image_size=(78, 78)):
    print("Starting dataset creation...")
    os.makedirs(output_dir, exist_ok=True)

    folders = {
        'MAA': os.path.join(base_path, 'R0002', 'MAA'),
        'Duramycin': os.path.join(base_path, 'R0002', 'Duramycin'),
        'HMPAO': os.path.join(base_path, 'R0002', 'HMPAO'),
    }

    image_files = {key: glob.glob(os.path.join(folder, '*.img')) for key, folder in folders.items()}

    print("Calculating averages for Duramycin and HMPAO images...")
    average_du = average_images(image_files['Duramycin'], image_size)
    average_hmpao = average_images(image_files['HMPAO'], image_size)

    percentages = np.linspace(0.5, 0.25, 10)
    
    for i, maa_file in enumerate(image_files['MAA']):
        print(f"Processing MAA image {i+1}/{len(image_files['MAA'])}")
        maa_image = read_and_process_image(maa_file, image_size)
        create_and_save_thresholded_images(maa_image, average_du, average_hmpao, percentages, output_dir, i)

    print("Processing completed successfully!")
    
if __name__ == "__main__":
    base_path = "."
    output_dir = "ThresholdedImages"
    
    create_lung_roi_dataset(base_path, output_dir)