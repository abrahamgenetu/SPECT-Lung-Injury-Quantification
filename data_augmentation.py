import os
import cv2
import albumentations as A
from glob import glob
from tqdm import tqdm

# Define paths for input and output folders
input_images_dir = "Dataset_Generated/img/"      # Path to folder with original images
input_masks_dir = "Dataset_Generated/mask/"        # Path to folder with original masks
output_images_dir = "Dataset_Generated/img"  # Path to save augmented images
output_masks_dir = "Dataset_Generated/mask"    # Path to save augmented masks

# Create output directories if they don't exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

# Define augmentations
transform = A.Compose([
    A.RandomRotate90(),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
], additional_targets={'mask': 'mask'})

# Get list of images and masks
image_paths = sorted(glob(os.path.join(input_images_dir, "*.png")))  # Adjust extension if needed
mask_paths = sorted(glob(os.path.join(input_masks_dir, "*.png")))

# Loop through each image-mask pair and apply augmentation
for i, (image_path, mask_path) in enumerate(tqdm(zip(image_paths, mask_paths), total=len(image_paths))):
    # Read the image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask as grayscale

    # Resize the mask to match image dimensions if they differ
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply augmentation
    augmented = transform(image=image, mask=mask)
    aug_image = augmented["image"]
    aug_mask = augmented["mask"]

    # Save augmented image and mask
    image_name = f"aug{i:03d}.png"
    mask_name = f"aug{i:03d}.png"
    cv2.imwrite(os.path.join(output_images_dir, image_name), aug_image)
    cv2.imwrite(os.path.join(output_masks_dir, mask_name), aug_mask)

print("Augmentation and saving completed.")