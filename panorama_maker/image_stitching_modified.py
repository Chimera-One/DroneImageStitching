import cv2
import os
import glob
import re
import yaml
import torch
import numpy as np
from lightglue import LightGlue, SuperPoint, ALIKED
from lightglue.utils import rbd
from torchvision import transforms

from concurrent.futures import ThreadPoolExecutor

def load_config(config_path):
    """Load configuration from a YAML file and compile regex patterns."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Adjust device setting based on the string from config
    config["device"] = torch.device(config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu")
    print("Configuration loaded successfully.")
    return config

def get_image_paths(config):
    """
    Get sorted image and mask paths based on timestamps from filenames.
    Returns:
        tuple: (sorted_image_paths, sorted_mask_paths)
    """
    # Load file lists
    all_image_files = glob.glob(os.path.join(config["image_directory"], '*.*'))
    
    # Note: Changed to look for 'mask_directory' to distinguish from images
    # If your masks are in the same folder, ensure the glob pattern distinguishes them
    all_mask_image_files = glob.glob(os.path.join(config["mask_image_directory"], '*.*'))
    all_heatmap_image_files = glob.glob(os.path.join(config["heatmap_image_directory"], '*.*'))

    def sort_by_filename(file_list):
        """Helper to sort files alphabetically by their filename."""
        return sorted(file_list, key=lambda x: os.path.basename(x))

    # Process both lists
    sorted_images = sort_by_filename(all_image_files)
    sorted_masks = sort_by_filename(all_mask_image_files)
    sorted_heatmaps = sort_by_filename(all_heatmap_image_files)

    # Optional: Safety check to ensure lists are aligned
    if len(sorted_images) != len(sorted_masks) or len(sorted_masks) != len(sorted_heatmaps):
        print(f"Warning: Found {len(sorted_images)} images but {len(sorted_masks)} masks and {len(sorted_heatmaps)} heapmaps.")
    else:
        print(f"Found {len(sorted_images)} images and matching masks to process.")

    return sorted_images, sorted_masks, sorted_heatmaps

    # Sort the image files based on timestamp
    image_files_with_timestamps.sort(key=lambda x: x[0])
    return [filepath for _, filepath in image_files_with_timestamps]

def extract_timestamp(filename):
    """Extract the timestamp from the filename assuming it is a number before the file extension."""
    basename = os.path.basename(filename)
    # Use regex to find the last sequence of digits before the file extension
    match = re.search(r'(\d+)(?=\.\w+$)', basename)
    if match:
        return int(match.group(1))
    return None

def process_image(path, config):
    """Load, crop, resize, and extract features from an image."""
    image_cv = cv2.imread(path)
    print(f"Processing image: {path}")
    if image_cv is None:
        print(f"Error loading image {path}")
        return None, None, None

    # Crop image
    h, w = image_cv.shape[:2]
    image_cropped = image_cv[config["top_crop"]:h - config["bottom_crop"], config["left_crop"]:w - config["right_crop"]]

    # Update size after cropping
    h_cropped, w_cropped = image_cropped.shape[:2]

    # Convert cropped image to tensor
    image_tensor = transforms.ToTensor()(image_cropped).to(config["device"]).unsqueeze(0)

    # Resize image for feature extraction
    image_resized = cv2.resize(image_cropped, (config["fixed_width"], config["fixed_height"]), interpolation=cv2.INTER_LINEAR)
    image_resized_tensor = transforms.ToTensor()(image_resized).to(config["device"]).unsqueeze(0)

    # Extract features
    if config['extractor'] == 'superpoint':
        print('Using SuperPoint for feature extraction')
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(config["device"])
    if config['extractor'] == 'aliked':
        print('Using ALIKED for feature extraction')
        extractor = ALIKED(max_num_keypoints=2048).eval().to(config["device"])

    with torch.no_grad():
        feats = extractor.extract(image_resized_tensor)
        feats['keypoints'] = feats['keypoints'].unsqueeze(0)
        feats['descriptors'] = feats['descriptors'].unsqueeze(0)
        feats['scores'] = feats.get('scores', torch.ones((1, feats['keypoints'].shape[1]), device=feats['keypoints'].device))
        feats = rbd(feats)

    print(f"Extracted features from image: {path}")
    return image_tensor, (h_cropped, w_cropped), feats

def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a NumPy image."""
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return image


def match_keypoints(feats_list, original_sizes, config):
    """Match keypoints between consecutive images and estimate transformations."""
    print(f"Matching keypoints for {len(feats_list)} images...")
    
    # Define matching network
    matcher = LightGlue(features=config['extractor']).eval().to(config["device"])

    accumulated_H = np.eye(3)
    transformations = [accumulated_H.copy()]
    all_corners = []

    total_matches = len(feats_list) - 1
    match_count = 0

    for i in range(1, len(feats_list)):
        match_count += 1
        print(f"Matching keypoints between image {i - 1} and image {i} ({match_count}/{total_matches})")
        feats0, feats1 = feats_list[i - 1], feats_list[i]
        matches_input = {"image0": feats0, "image1": feats1}
        matches01 = matcher(matches_input)
        matches01_rbd = rbd(matches01)
        print('matches01', matches01)

        kpts0 = feats0["keypoints"].squeeze(0)
        kpts1 = feats1["keypoints"].squeeze(0)
        matches = matches01_rbd["matches0"].squeeze(0)

        # Filter valid matches
        valid_matches = matches > -1
        m_kpts0 = kpts0[valid_matches]
        m_kpts1 = kpts1[matches[valid_matches]]

        # Scale keypoints back to original image dimensions
        h_resized, w_resized = config["fixed_height"], config["fixed_width"]
        h0_orig, w0_orig = original_sizes[i - 1]
        hi_orig, wi_orig = original_sizes[i]

        scale_x0 = w0_orig / w_resized
        scale_y0 = h0_orig / h_resized
        scale_xi = wi_orig / w_resized
        scale_yi = hi_orig / h_resized

        m_kpts0_orig = m_kpts0.clone()
        m_kpts0_orig[:, 0] *= scale_x0
        m_kpts0_orig[:, 1] *= scale_y0

        m_kpts1_orig = m_kpts1.clone()
        m_kpts1_orig[:, 0] *= scale_xi
        m_kpts1_orig[:, 1] *= scale_yi

        m_kpts0_np = m_kpts0_orig.cpu().numpy()
        m_kpts1_np = m_kpts1_orig.cpu().numpy()

        if len(m_kpts0_np) >= 3:
            H_affine, inliers = cv2.estimateAffinePartial2D(m_kpts1_np, m_kpts0_np, method=cv2.RANSAC) #can also be 'LMEDS'
            if H_affine is not None:
                H = np.vstack([H_affine, [0, 0, 1]])
                accumulated_H = accumulated_H @ H
                transformations.append(accumulated_H.copy())

                hi, wi = original_sizes[i]
                corners_i = np.array([[0, 0], [0, hi], [wi, hi], [wi, 0]], dtype=np.float32)
                rotation_translation = accumulated_H[:2, :]
                corners_i_transformed = cv2.transform(corners_i.reshape(-1, 1, 2), rotation_translation)
                all_corners.append(corners_i_transformed.reshape(-1, 2))
                print(f"Transformation found between image {i - 1} and image {i}.")
            else:
                print(f"Transformation failed between image {i-1} and image {i}.")
        else:
            print(f"Not enough matches between image {i-1} and image {i}.")

    return transformations, all_corners

def stitch_images(original_images, mask_images, heatmap_images, transformations, all_corners):
    """Stitch images together into a panorama using original blending logic."""
    print("Starting stitching of images...")
    all_corners = np.vstack(all_corners)
    x_min, y_min = np.int32(all_corners.min(axis=0) - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0) + 0.5)
    panorama_width = x_max - x_min
    panorama_height = y_max - y_min

    total_stitching = len(original_images)
    stitch_count = 0

    # Compute the translation matrix
    translation = np.array([[1, 0, -x_min],
                            [0, 1, -y_min],
                            [0, 0, 1]])

    # Initialize the panorama and mask
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    panorama_mask = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    panorama_heatmap = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    mask_panorama = np.zeros((panorama_height, panorama_width), dtype=np.uint8)

    # Warp and blend each image using the original blending logic
    for i in range(len(original_images)):
        stitch_count += 1
        print(f"Stitching image {stitch_count}/{total_stitching}")

        H = transformations[i]
        # Compute the total transformation
        H_total = translation @ H

        # Warp the image
        image_i_np = tensor_to_image(original_images[i])
        warped_image_i = cv2.warpPerspective(image_i_np, H_total, (panorama_width, panorama_height))
        warped_mask_image_i = cv2.warpPerspective(mask_images[i], H_total, (panorama_width, panorama_height))
        warped_heatmap_image_i = cv2.warpPerspective(heatmap_images[i], H_total, (panorama_width, panorama_height))

        # Warp the mask
        hi, wi = image_i_np.shape[:2]
        mask_i = np.ones((hi, wi), dtype=np.uint8) * 255
        warped_mask_i = cv2.warpPerspective(mask_i, H_total, (panorama_width, panorama_height))

        # Update panorama and mask_panorama with original blending logic
        mask_overlap = warped_mask_i > 0
        panorama[mask_overlap] = warped_image_i[mask_overlap]
        panorama_mask[mask_overlap] = warped_mask_image_i[mask_overlap]
        panorama_heatmap[mask_overlap] = warped_heatmap_image_i[mask_overlap]

        # Mask for masking panorama
        mask_panorama[mask_overlap] = warped_mask_i[mask_overlap]
        print(f"Stitching image {i + 1} of {len(original_images)}")

    print("Stitching completed.")
    return panorama, panorama_mask, panorama_heatmap

def run_stitching_pipeline(config_path, save_img=False):
    """Run the complete stitching pipeline with configuration from a YAML file."""
    print("Running image stitching pipeline...")
    config = load_config(config_path)
    
    # 1. Get paths for both images and masks
    image_paths, mask_paths, heatmap_paths = get_image_paths(config)

    print("Processing images and loading masks...")
    total_images = len(image_paths)
    processed_count = 0

    original_images = []
    original_sizes = []
    feats_list = []
    mask_images = []
    heatmap_images = []

    # 2. Process images in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        # executor.map preserves the order of input 'image_paths'
        results = executor.map(lambda path: process_image(path, config), image_paths)

    # 3. Iterate through results AND mask_paths together to ensure alignment
    for (image_tensor, size, feats), mask_path, heatmap_path in zip(results, mask_paths, heatmap_paths):
        processed_count += 1
        print(f"Processed {processed_count}/{total_images} images.")
        
        if image_tensor is not None:
            # Load the mask as a numpy array
            # using imread to get the np array (default BGR)
            # Add cv2.IMREAD_GRAYSCALE if your stitching pipeline expects 1-channel masks
            mask_np = cv2.imread(mask_path)
            heatmap_np = cv2.imread(heatmap_path)
            
            if mask_np is None:
                print(f"Warning: Could not load mask at {mask_path}. Skipping corresponding image.")
                continue

            if heatmap_np is None:
                print(f"Warning: Could not load heatmap at {heatmap_path}. Skipping corresponding image.")
                continue

            # Only append data if both image and mask are valid
            original_images.append(image_tensor)
            original_sizes.append(size)
            feats_list.append(feats)
            mask_images.append(mask_np)
            heatmap_images.append(heatmap_np)

    print("Matching keypoints and estimating transformations...")
    transformations, all_corners = match_keypoints(feats_list, original_sizes, config)
    
    # 4. Pass the list of np arrays (mask_images) instead of paths
    panorama, panorama_mask, panorama_heatmap = stitch_images(original_images, mask_images, heatmap_images, transformations, all_corners)
    
    print("Pipeline completed.")

    if config['save_image']:
        cv2.imwrite(os.path.join(config['output_dir'], config['output_filename']) + ".png", panorama)
        cv2.imwrite(os.path.join(config['output_dir'], config['output_filename']) + "_mask.png", panorama_mask)
        cv2.imwrite(os.path.join(config['output_dir'], config['output_filename']) + "_heatmap.png", panorama_heatmap)

    return panorama, panorama_mask, panorama_heatmap

# Use this guard to prevent auto-execution when imported
if __name__ == "__main__":
    config_path = "config.yaml"
    panorama, panorama_mask, panorama_heatmap = run_stitching_pipeline(config_path)
