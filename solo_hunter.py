import os
import glob
import cv2
import numpy as np
import albumentations as A
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ---------------------------
# Configuration and Paths
# ---------------------------
# Folder where your raw images are stored
SOURCE_FOLDER = r"C:\Wild\eagle_eyes_dataset"
# Folder to save augmented images
AUG_FOLDER = r"C:\Wild\augmented"
# Folder to save final output with bounding boxes
OUTPUT_FOLDER = r"C:\Wild\output"

# Create folders if they do not exist
os.makedirs(AUG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------
# Data Augmentation Function
# ---------------------------
def augment_images(source_folder, aug_folder):
    # Define an augmentation pipeline using albumentations
    aug_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        # Add more augmentations if needed
    ])
    
    # Supported image file extensions
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(source_folder, ext)))
    
    augmented_image_paths = []
    print(f"Found {len(image_paths)} images for augmentation.")
    
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue
        # Apply augmentations
        augmented = aug_pipeline(image=image)
        aug_image = augmented['image']
        # Save augmented image using the original base name with a suffix
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        new_filename = f"{base_name}_aug.jpg"
        save_path = os.path.join(aug_folder, new_filename)
        cv2.imwrite(save_path, aug_image)
        augmented_image_paths.append(save_path)
    
    print(f"Augmented images saved to {aug_folder}.")
    return augmented_image_paths

# ---------------------------
# Tiling (Grid Division) Function
# ---------------------------
def tile_image(image, grid_rows=3, grid_cols=3):
    """
    Divide an image into grid_rows x grid_cols tiles.
    Returns a list of tuples: (tile_image, (start_x, start_y))
    where (start_x, start_y) is the top-left corner of the tile in the original image.
    """
    img_h, img_w = image.shape[:2]
    tile_h = img_h // grid_rows
    tile_w = img_w // grid_cols
    tiles = []
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            start_y = i * tile_h
            start_x = j * tile_w
            # For the last row/column, take any remaining pixels
            end_y = img_h if i == grid_rows - 1 else start_y + tile_h
            end_x = img_w if j == grid_cols - 1 else start_x + tile_w
            tile = image[start_y:end_y, start_x:end_x]
            tiles.append((tile, (start_x, start_y)))
    return tiles

# ---------------------------
# Detection on a Tile
# ---------------------------
def run_detection_on_tile(model, tile, tile_origin, conf_threshold=0.25):
    """
    Run detection on one tile image and adjust bounding box coordinates.
    :param model: YOLO model
    :param tile: Tile image (numpy array)
    :param tile_origin: (start_x, start_y) coordinate of the tile in the original image.
    :param conf_threshold: confidence threshold for detections.
    :return: list of adjusted detections [[x1, y1, x2, y2, conf, class], ...]
    """
    results = model.predict(source=tile, conf=conf_threshold, verbose=False)
    detections = []
    if results and len(results) > 0:
        # Loop through each detection from the first result
        for det in results[0].boxes:
            # Get bounding box [x1, y1, x2, y2]
            box = det.xyxy[0].cpu().numpy()
            conf = float(det.conf[0].item())
            cls = int(det.cls[0].item())
            start_x, start_y = tile_origin
            # Adjust coordinates relative to the original image
            x1, y1, x2, y2 = box
            detections.append([x1 + start_x, y1 + start_y, x2 + start_x, y2 + start_y, conf, cls])
    return detections

# ---------------------------
# Process Full Image and Merge Detections
# ---------------------------
def process_full_image(model, image_path, grid_rows=3, grid_cols=3, conf_threshold=0.25):
    """
    Process a full image by tiling it, running detection on each tile, and merging detections.
    Returns the original image and the list of detections.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Cannot read image {image_path}")
        return None, []
    
    tiles = tile_image(original_image, grid_rows, grid_cols)
    all_detections = []
    for tile, origin in tiles:
        dets = run_detection_on_tile(model, tile, origin, conf_threshold)
        all_detections.extend(dets)
    return original_image, all_detections

# ---------------------------
# Draw Bounding Boxes on Image
# ---------------------------
def draw_detections(image, detections):
    """
    Draw bounding boxes on the image.
    Each detection is a list: [x1, y1, x2, y2, conf, class]
    """
    for (x1, y1, x2, y2, conf, cls) in detections:
        # Drawing rectangle and confidence
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{conf:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# ---------------------------
# Main Workflow
# ---------------------------
def main():
    # Step 1: Augment images and save them to the augmented folder.
    augmented_images = augment_images(SOURCE_FOLDER, AUG_FOLDER)
    
    if not augmented_images:
        print("No augmented images found. Exiting.")
        return

    # Step 2: Load YOLOv8 model and move it to GPU.
    print("Loading YOLOv8 model (yolov8s.pt)...")
    model = YOLO('yolov8s.pt')  # Use the small model or replace with your custom model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}.")

    # Process each augmented image: tile, detect, draw detections, and save output.
    for img_path in augmented_images:
        print(f"Processing image: {img_path}")
        orig_img, detections = process_full_image(model, img_path, grid_rows=3, grid_cols=3, conf_threshold=0.25)
        if orig_img is None:
            continue
        # Draw detections on the original image
        result_img = draw_detections(orig_img, detections)
        # Save the result with a suffix in the output folder
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_detected.jpg")
        cv2.imwrite(out_path, result_img)
        print(f"Detection result saved to: {out_path}")

    # Optional: Display one result using matplotlib for quick verification.
    sample_output = os.path.join(OUTPUT_FOLDER, os.listdir(OUTPUT_FOLDER)[0])
    img_to_show = cv2.imread(sample_output)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB))
    plt.title("Sample Detection Result")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()