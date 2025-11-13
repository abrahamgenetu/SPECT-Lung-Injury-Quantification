"""
DICOM/Image Processing Tools for SPECT Lung Injury Analysis
Handles .img file format, image subtraction, ROI analysis, and batch processing

Based on lab protocol: MAA - Duramycin - HMPAO subtraction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import json

class SPECTImageLoader:
    """Load and process .img format SPECT images"""
    
    def __init__(self, image_size: Tuple[int, int] = (78, 78)):
        self.m, self.n = image_size
        
    def load_img_file(self, filepath: str) -> np.ndarray:
        """
        Load a single .img file (binary float format)
        
        Args:
            filepath: Path to .img file
            
        Returns:
            2D numpy array (rotated and transposed to match MATLAB)
        """
        with open(filepath, 'rb') as f:
            # Read as little-endian float32
            data = np.fromfile(f, dtype='<f4', count=self.m * self.n)
            
        # Reshape to matrix
        image = data.reshape(self.m, self.n)
        
        # Transpose and rotate 270 degrees (equivalent to MATLAB operations)
        image = image.T
        image = np.rot90(image, k=3)  # 270 degree rotation
        
        return image
    
    def load_folder(self, folder_path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Load all .img files from folder and compute average
        
        Args:
            folder_path: Path to folder containing .img files
            
        Returns:
            (average_image, list_of_individual_images)
        """
        folder = Path(folder_path)
        img_files = sorted(folder.glob('*.img'))
        
        if len(img_files) == 0:
            raise ValueError(f"No .img files found in {folder_path}")
        
        print(f"Found {len(img_files)} images in {folder_path}")
        
        images = []
        image_sum = np.zeros((self.m, self.n))
        
        for img_file in img_files:
            image = self.load_img_file(str(img_file))
            images.append(image)
            image_sum += image
        
        average_image = image_sum / len(img_files)
        
        return average_image, images

class ROISelector:
    """Interactive ROI selection using matplotlib"""
    
    def __init__(self, image: np.ndarray):
        self.image = image
        self.roi_mask = None
        self.rect_coords = None
        
    def select_roi_interactive(self) -> np.ndarray:
        """
        Interactive ROI selection with rectangle tool
        
        Returns:
            Binary mask of selected ROI
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.image, cmap='hot')
        ax.set_title('Draw ROI on the image (Press Enter when done)')
        ax.axis('image')
        
        # Initial rectangle position
        x0, y0, width, height = 25, 15, 25, 20
        
        # Create rectangle selector
        self.rect_selector = RectangleSelector(
            ax, self._on_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        
        # Add initial rectangle
        rect = Rectangle((x0, y0), width, height, 
                        linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        self.rect_coords = (x0, y0, width, height)
        
        # Connect Enter key
        fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        print("Draw rectangle ROI and press Enter to continue...")
        plt.show()
        
        # Create mask from rectangle
        if self.rect_coords is not None:
            x, y, w, h = self.rect_coords
            self.roi_mask = np.zeros_like(self.image, dtype=bool)
            self.roi_mask[int(y):int(y+h), int(x):int(x+w)] = True
        
        return self.roi_mask
    
    def _on_select(self, eclick, erelease):
        """Callback for rectangle selection"""
        self.rect_coords = (
            min(eclick.xdata, erelease.xdata),
            min(eclick.ydata, erelease.ydata),
            abs(erelease.xdata - eclick.xdata),
            abs(erelease.ydata - eclick.ydata)
        )
    
    def _on_key(self, event):
        """Callback for key press"""
        if event.key == 'enter':
            plt.close()

class ThresholdSegmentation:
    """Histogram-based thresholding and segmentation"""
    
    def __init__(self, image: np.ndarray, roi_mask: np.ndarray):
        self.image = image
        self.roi_mask = roi_mask
        self.segmented_mask = None
        
    def calculate_histogram(self, num_bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate histogram of ROI pixel values"""
        roi_values = self.image[self.roi_mask]
        
        # Convert to uint8 range
        roi_values_normalized = ((roi_values - roi_values.min()) / 
                                (roi_values.max() - roi_values.min()) * 255)
        roi_values_uint8 = roi_values_normalized.astype(np.uint8)
        
        hist, bins = np.histogram(roi_values_uint8, bins=num_bins, range=(0, 255))
        
        return hist, bins
    
    def threshold_segmentation(self, percentage: float = 0.5) -> np.ndarray:
        """
        Apply threshold segmentation based on percentage of max intensity
        
        Args:
            percentage: Fraction of max intensity (0-1)
            
        Returns:
            Binary segmentation mask
        """
        # Get ROI values
        roi_values = self.image[self.roi_mask]
        max_intensity = np.max(roi_values)
        threshold_value = max_intensity * percentage
        
        print(f"Maximum intensity in ROI: {max_intensity:.4f}")
        print(f"Threshold value ({percentage*100}%): {threshold_value:.4f}")
        
        # Create segmentation mask
        self.segmented_mask = (self.image >= threshold_value) & self.roi_mask
        
        return self.segmented_mask
    
    def crop_bottom_roi(self, crop_fraction: float = 0.25) -> np.ndarray:
        """
        Crop bottom portion of ROI to exclude liver
        
        Args:
            crop_fraction: Fraction of ROI height to remove from bottom
            
        Returns:
            Cropped binary mask
        """
        if self.segmented_mask is None:
            raise ValueError("Must run threshold_segmentation first")
        
        # Find ROI boundaries
        rows, cols = np.where(self.segmented_mask)
        if len(rows) == 0:
            return self.segmented_mask
        
        top = rows.min()
        bottom = rows.max()
        roi_height = bottom - top + 1
        
        # Calculate crop amount
        to_crop = int(crop_fraction * roi_height)
        
        # Create cropped mask
        cropped_mask = self.segmented_mask.copy()
        cropped_mask[bottom - to_crop + 1:, :] = False
        
        print(f"ROI height: {roi_height} pixels")
        print(f"Cropped {to_crop} pixels from bottom ({crop_fraction*100}%)")
        
        return cropped_mask
    
    def visualize_histogram_threshold(self, percentage: float = 0.5):
        """Visualize histogram with threshold"""
        hist, bins = self.calculate_histogram()
        
        roi_values = self.image[self.roi_mask]
        roi_values_normalized = ((roi_values - roi_values.min()) / 
                                (roi_values.max() - roi_values.min()) * 255)
        max_uint8 = np.max(roi_values_normalized.astype(np.uint8))
        threshold_uint8 = max_uint8 * percentage
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot full histogram
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(bin_centers, hist, width=1.0, color='blue', alpha=0.7, label='Histogram')
        
        # Highlight threshold range
        threshold_range = bin_centers <= threshold_uint8
        ax.bar(bin_centers[threshold_range], hist[threshold_range], 
              width=1.0, color='red', alpha=0.7, 
              label=f'Threshold Range (≤{percentage*100}% of max)')
        
        ax.set_xlabel('Pixel Intensity (normalized)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Histogram with Threshold Range Based on Max Intensity', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class MultiTracerAnalysis:
    """Complete multi-tracer SPECT analysis pipeline"""
    
    def __init__(self, maa_folder: str, du_folder: str, hmpao_folder: str):
        self.loader = SPECTImageLoader()
        
        # Load all images
        print("\nLoading MAA images...")
        self.avg_maa, self.maa_images = self.loader.load_folder(maa_folder)
        
        print("\nLoading Duramycin images...")
        self.avg_du, self.du_images = self.loader.load_folder(du_folder)
        
        print("\nLoading HMPAO images...")
        self.avg_hmpao, self.hmpao_images = self.loader.load_folder(hmpao_folder)
        
        # Calculate difference images
        self.diff1 = self.avg_maa - self.avg_du
        self.diff2 = self.avg_maa - self.avg_du - self.avg_hmpao
        
        self.roi_mask = None
        self.cropped_mask = None
        self.results = {}
        
    def visualize_averages(self):
        """Display average images and differences"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(self.avg_maa, cmap='hot')
        axes[0].set_title('Average MAA Image', fontsize=13, fontweight='bold')
        axes[0].axis('image')
        
        axes[1].imshow(self.diff1, cmap='hot')
        axes[1].set_title('MAA - Duramycin', fontsize=13, fontweight='bold')
        axes[1].axis('image')
        
        axes[2].imshow(self.diff2, cmap='hot')
        axes[2].set_title('MAA - Duramycin - HMPAO', fontsize=13, fontweight='bold')
        axes[2].axis('image')
        
        plt.tight_layout()
        plt.show()
    
    def perform_roi_selection(self, use_diff2: bool = True):
        """Interactive ROI selection"""
        image = self.diff2 if use_diff2 else self.diff1
        
        selector = ROISelector(image)
        self.roi_mask = selector.select_roi_interactive()
        
        return self.roi_mask
    
    def perform_threshold_segmentation(self, percentage: float = 0.5, 
                                      crop_bottom: bool = True,
                                      crop_fraction: float = 0.25):
        """Apply threshold segmentation with optional cropping"""
        if self.roi_mask is None:
            raise ValueError("Must select ROI first")
        
        segmenter = ThresholdSegmentation(self.diff2, self.roi_mask)
        
        # Show histogram
        segmenter.visualize_histogram_threshold(percentage)
        
        # Threshold
        segmented = segmenter.threshold_segmentation(percentage)
        
        # Crop if requested
        if crop_bottom:
            self.cropped_mask = segmenter.crop_bottom_roi(crop_fraction)
        else:
            self.cropped_mask = segmented
        
        # Visualize segmentation
        self._visualize_segmentation(segmented, self.cropped_mask)
        
        return self.cropped_mask
    
    def _visualize_segmentation(self, uncropped_mask: np.ndarray, 
                               cropped_mask: np.ndarray):
        """Visualize original and cropped ROI boundaries"""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.diff2, cmap='hot')
        
        # Draw uncropped boundaries in black
        from skimage.segmentation import find_boundaries
        uncropped_boundary = find_boundaries(uncropped_mask, mode='outer')
        ax.contour(uncropped_boundary, colors='black', linewidths=2)
        
        # Draw cropped boundaries in red
        cropped_boundary = find_boundaries(cropped_mask, mode='outer')
        ax.contour(cropped_boundary, colors='red', linewidths=2)
        
        ax.set_title('Original Image with ROI Overlays', fontsize=14, fontweight='bold')
        ax.axis('image')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='Uncropped ROI'),
            Line2D([0], [0], color='red', lw=2, label='Cropped ROI (Final)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # Display binary mask
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cropped_mask, cmap='gray')
        ax.set_title('Binary Mask Image of the ROI', fontsize=14, fontweight='bold')
        ax.axis('image')
        plt.tight_layout()
        plt.show()
    
    def analyze_duramycin_intensities(self) -> Dict:
        """
        Calculate mean intensity for each Duramycin image within cropped ROI
        
        Returns:
            Dictionary with individual and average intensities
        """
        if self.cropped_mask is None:
            raise ValueError("Must perform segmentation first")
        
        print("\n" + "="*60)
        print("DURAMYCIN INTENSITY ANALYSIS")
        print("="*60)
        
        du_intensities = []
        
        for i, du_image in enumerate(self.du_images):
            # Apply mask
            segmented_region = self.cropped_mask * du_image
            mean_intensity = np.mean(du_image[self.cropped_mask])
            du_intensities.append(mean_intensity)
            
            print(f"Image {i+1}: Mean Intensity = {mean_intensity:.4f}")
            
            # Visualize overlay
            if i < 3:  # Show first 3 images
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(du_image, cmap='hot')
                
                # Overlay segmentation boundary
                from skimage.segmentation import find_boundaries
                boundary = find_boundaries(self.cropped_mask, mode='outer')
                ax.contour(boundary, colors='red', linewidths=2)
                
                ax.set_title(f'Duramycin Image {i+1} with MAA ROI Overlay\nMean Intensity: {mean_intensity:.4f}',
                           fontsize=13, fontweight='bold')
                ax.axis('image')
                plt.tight_layout()
                plt.show()
        
        average_intensity = np.mean(du_intensities)
        std_intensity = np.std(du_intensities)
        
        print("\n" + "="*60)
        print(f"AVERAGE DURAMYCIN INTENSITY: {average_intensity:.4f} ± {std_intensity:.4f}")
        print("="*60)
        
        self.results = {
            'individual_intensities': du_intensities,
            'mean_intensity': average_intensity,
            'std_intensity': std_intensity,
            'n_images': len(du_intensities),
            'roi_area_pixels': np.sum(self.cropped_mask),
            'roi_volume_ml': np.sum(self.cropped_mask) * (2.21/10)**2 * (2.21/10) / 1000
        }
        
        return self.results
    
    def analyze_all_tracers(self) -> pd.DataFrame:
        """Analyze all three tracers with the same ROI"""
        if self.cropped_mask is None:
            raise ValueError("Must perform segmentation first")
        
        results_df = pd.DataFrame()
        
        for tracer_name, images in [('MAA', self.maa_images), 
                                     ('Duramycin', self.du_images),
                                     ('HMPAO', self.hmpao_images)]:
            intensities = []
            for image in images:
                mean_intensity = np.mean(image[self.cropped_mask])
                intensities.append(mean_intensity)
            
            temp_df = pd.DataFrame({
                'Tracer': tracer_name,
                'Image_Number': range(1, len(intensities) + 1),
                'Mean_Intensity': intensities
            })
            results_df = pd.concat([results_df, temp_df], ignore_index=True)
        
        return results_df
    
    def save_results(self, output_file: str = 'spect_analysis_results.json'):
        """Save analysis results to JSON"""
        if not self.results:
            print("No results to save")
            return
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = {
            'individual_intensities': [float(x) for x in self.results['individual_intensities']],
            'mean_intensity': float(self.results['mean_intensity']),
            'std_intensity': float(self.results['std_intensity']),
            'n_images': int(self.results['n_images']),
            'roi_area_pixels': int(self.results['roi_area_pixels']),
            'roi_volume_ml': float(self.results['roi_volume_ml'])
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\nResults saved to {output_file}")

def main():
    """Main execution function for multi-tracer analysis"""
    
    print("="*70)
    print("SPECT MULTI-TRACER ANALYSIS FOR RADIATION LUNG INJURY")
    print("Lab Protocol: MAA - Duramycin - HMPAO Subtraction")
    print("="*70)
    
    # Define folder paths (adjust to your data structure)
    base_folder = "R0002"
    maa_folder = f"{base_folder}/MAA"
    du_folder = f"{base_folder}/Duramycin"
    hmpao_folder = f"{base_folder}/HMPAO"
    
    try:
        # Initialize analysis
        print("\n1. Loading images...")
        analyzer = MultiTracerAnalysis(maa_folder, du_folder, hmpao_folder)
        
        # Visualize averages and differences
        print("\n2. Displaying average images and differences...")
        analyzer.visualize_averages()
        
        # ROI selection
        print("\n3. ROI selection (interactive)...")
        analyzer.perform_roi_selection(use_diff2=True)
        
        # Threshold segmentation
        print("\n4. Performing threshold segmentation...")
        percentage = 0.5  # Can be adjusted or made interactive
        analyzer.perform_threshold_segmentation(
            percentage=percentage,
            crop_bottom=True,
            crop_fraction=0.25
        )
        
        # Analyze Duramycin intensities
        print("\n5. Analyzing Duramycin intensities...")
        du_results = analyzer.analyze_duramycin_intensities()
        
        # Optional: Analyze all tracers
        print("\n6. Analyzing all tracers...")
        all_results = analyzer.analyze_all_tracers()
        print("\nAll Tracer Results:")
        print(all_results.groupby('Tracer')['Mean_Intensity'].agg(['mean', 'std']))
        
        # Save results
        print("\n7. Saving results...")
        analyzer.save_results('duramycin_analysis_results.json')
        all_results.to_csv('all_tracers_analysis.csv', index=False)
        print("All tracer results saved to 'all_tracers_analysis.csv'")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease adjust folder paths in the script to match your data structure.")
        print("Expected structure:")
        print("  R0002/")
        print("    ├── MAA/")
        print("    │   ├── image001.img")
        print("    │   └── ...")
        print("    ├── Duramycin/")
        print("    │   ├── image001.img")
        print("    │   └── ...")
        print("    └── HMPAO/")
        print("        ├── image001.img")
        print("        └── ...")

if __name__ == "__main__":
    main()