import os
import numpy as np
from PIL import Image
import glob
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

def read_image(filepath, image_size):
    img = np.fromfile(filepath, dtype=np.float32).reshape(image_size)
    img = np.rot90(img, k=3)
    return img

def average_images(file_list, image_size):
    if len(file_list) == 0:
        return None
    return np.mean([read_image(f, image_size) for f in file_list], axis=0)

def calculate_thresholds(image, n_thresholds):
    thresholds = []
    image_copy = image.copy()

    for _ in range(n_thresholds):
        if np.isnan(image_copy).any() or np.all(image_copy == 0):
            break
        threshold = threshold_otsu(image_copy)
        thresholds.append(threshold)
        image_copy[image_copy > threshold] = 0

    return thresholds

def create_masks(image_diff, thresholds):
    masks = []
    for threshold in thresholds:
        mask = (image_diff >= threshold).astype(np.uint8) * 255
        mask = gaussian_filter(mask, sigma=1.6) > 128
        masks.append(mask.astype(np.uint8) * 255)
    return masks

def mutual_information(mask1, mask2):
    hist, _, _ = np.histogram2d(mask1.ravel(), mask2.ravel(), bins=20)
    hist = hist / np.sum(hist)
    p_x = np.sum(hist, axis=1, keepdims=True)
    p_y = np.sum(hist, axis=0, keepdims=True)
    p_xy_indep = p_x * p_y
    nz = (hist > 0) & (p_xy_indep > 0)
    return np.sum(hist[nz] * np.log(hist[nz] / p_xy_indep[nz]))

def save_best_mask(masks, output_dir, rat_id, img_id):
    if len(masks) == 0:
        print(f"  No valid masks for {rat_id} image {img_id}")
        return
    ref_mask = masks[0]
    best_mask = max(masks, key=lambda m: mutual_information(ref_mask, m))
    out_path = os.path.join(output_dir, "mask", f"{rat_id}_{img_id:03d}.png")
    Image.fromarray(best_mask).save(out_path)

def save_diff_image(image_diff, output_dir, rat_id, img_id):
    plt.figure()
    plt.imshow(image_diff, cmap="jet")
    plt.axis("off")
    out_path = os.path.join(output_dir, "img",  f"{rat_id}_{img_id:03d}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def process_images(base_path, output_dir, image_size=(78, 78), n_thresholds=10):
    os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "img"), exist_ok=True)

    train_rats = ["R0001", "R0002", "R0003", "R0004", "R0005", "R0006", "R0007", "R0008","R0009", "R0010", "R0011", "R0012", "R0013", "R0014", "R0015", "R0016","R0017", "R0018"]
    test_rats = ["R017", "R018", "R019", "R020", "R021", "R022", "R023", "R024", "R025"]


    for rat_id in test_rats:
        print(f"Processing {rat_id}...")

        folders = {
            "MAA": os.path.join(base_path, rat_id, "MAA"),
            "Duramycin": os.path.join(base_path, rat_id, "Duramycin"),
            "HMPAO": os.path.join(base_path, rat_id, "HMPAO")
        }

        image_files = {
            k: glob.glob(os.path.join(v, "*.img"))
            for k, v in folders.items()
        }

        # Check for missing required files
        if not image_files["MAA"]:
            print(f"  No MAA images for {rat_id}. Skipping.")
            continue
        if not image_files["Duramycin"] or not image_files["HMPAO"]:
            print(f"  Missing Duramycin or HMPAO for {rat_id}. Skipping.")
            continue

        avg_duramycin = average_images(image_files["Duramycin"], image_size)
        avg_hmpao = average_images(image_files["HMPAO"], image_size)

        if avg_duramycin is None or avg_hmpao is None:
            print(f"  Failed to compute averages for {rat_id}. Skipping.")
            continue

        for i, maa_file in enumerate(image_files["MAA"]):
            maa_image = read_image(maa_file, image_size)
            du_image = avg_duramycin + maa_image
            image_diff = maa_image - avg_duramycin - avg_hmpao

            # sanity check to avoid NaN Otsu crash
            if np.isnan(image_diff).any() or not np.isfinite(image_diff).any():
                print(f"  Skipping {rat_id} image {i}: NaN or invalid image_diff")
                continue

            # compute thresholds safely
            thresholds = calculate_thresholds(image_diff.copy(), n_thresholds)
            if len(thresholds) == 0:
                print(f"  No thresholds computed for {rat_id} image {i}. Skipping.")
                continue

            masks = create_masks(image_diff, thresholds)

            save_diff_image(du_image, output_dir, rat_id, i)
            save_best_mask(masks, output_dir, rat_id, i)

if __name__ == "__main__":
    base_path = "Raw_dataset"
    output_dir = "Test_Dataset_Generated"
    process_images(base_path, output_dir)
    print("Dataset Generated Successfully!")
