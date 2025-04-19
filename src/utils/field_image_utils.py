import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob
import random
import json
from pathlib import Path
import cv2

from .paths import ensure_dir_exists

# Directories for field image dataset
def get_field_image_paths(base_dir):
    """Get paths for the field image dataset"""
    field_images_dir = os.path.join(base_dir, 'field_images')
    field_labels_dir = os.path.join(base_dir, 'field_labels')
    field_masks_dir = os.path.join(base_dir, 'field_masks')
    field_bounds_dir = os.path.join(base_dir, 'field_bounds')
    field_stats_json = os.path.join(base_dir, 'field_stats.json')
    stats_json = os.path.join(base_dir, 'stats.json')
    
    return {
        'field_images_dir': field_images_dir,
        'field_labels_dir': field_labels_dir,
        'field_masks_dir': field_masks_dir,
        'field_bounds_dir': field_bounds_dir,
        'field_stats_json': field_stats_json,
        'stats_json': stats_json
    }

def load_field_stats(base_dir):
    """Load field stats from JSON file"""
    paths = get_field_image_paths(base_dir)
    
    with open(paths['stats_json'], 'r') as f:
        stats = json.load(f)
    
    return stats

def get_field_class_names(base_dir):
    """Get the names of all field classes from the directory structure"""
    paths = get_field_image_paths(base_dir)
    
    # Get all subdirectories in the field_labels directory
    class_names = [d for d in os.listdir(paths['field_labels_dir']) 
                  if os.path.isdir(os.path.join(paths['field_labels_dir'], d))]
    
    return class_names

def load_image_and_mask(image_path, mask_path):
    """Load an image and its corresponding mask"""
    image = Image.open(image_path)
    mask = Image.open(mask_path) if mask_path else None
    
    return np.array(image), np.array(mask) if mask else None

def get_corresponding_mask(image_filename, masks_dir):
    """Find the corresponding mask for an image based on its filename"""
    image_id = os.path.basename(image_filename).split('.')[0]  # Remove extension
    mask_path = os.path.join(masks_dir, f"{image_id}.png")
    
    if os.path.exists(mask_path):
        return mask_path
    return None

def get_image_label_pairs(base_dir, class_name, limit=None):
    """Get pairs of image and label paths for a specific class"""
    paths = get_field_image_paths(base_dir)
    
    # Find all label files for the specified class
    label_dir = os.path.join(paths['field_labels_dir'], class_name)
    if not os.path.exists(label_dir):
        return []
    
    label_files = glob.glob(os.path.join(label_dir, "*.png"))
    if limit:
        label_files = label_files[:limit]
    
    pairs = []
    for label_path in label_files:
        # Extract the filename (without extension)
        filename = os.path.basename(label_path).split('.')[0]
        
        # Find the corresponding RGB image
        rgb_dir = os.path.join(paths['field_images_dir'], 'rgb')
        image_path = os.path.join(rgb_dir, f"{filename}.jpg")
        
        # Find the corresponding mask
        mask_path = os.path.join(paths['field_masks_dir'], f"{filename}.png")
        
        if os.path.exists(image_path):
            pairs.append({
                'image_path': image_path,
                'label_path': label_path,
                'mask_path': mask_path if os.path.exists(mask_path) else None,
                'class_name': class_name
            })
    
    return pairs

def create_dataset(dataset_dir, output_dir, val_split=0.15, test_split=0.15, limit_per_class=None):
    """
    Create a dataset from field images and organize it into train/val/test splits
    
    Args:
        dataset_dir: Base directory containing the dataset
        output_dir: Directory to save the processed dataset
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        limit_per_class: Maximum number of samples per class (None for all)
        
    Returns:
        A dictionary with train/val/test counts and class distribution
    """
    field_images_dir = os.path.join(dataset_dir, 'field_images', 'rgb')  # RGB images directory
    field_masks_dir = os.path.join(dataset_dir, 'field_masks')
    field_labels_dir = os.path.join(dataset_dir, 'field_labels')
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(field_images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    print(f"Found {len(image_files)} image files in {field_images_dir}")
    
    # Get all class directories in field_labels
    class_dirs = [d for d in os.listdir(field_labels_dir) if os.path.isdir(os.path.join(field_labels_dir, d))]
    print(f"Found {len(class_dirs)} class directories: {class_dirs}")
    
    # Create class directories in train/val/test
    for split_dir in [train_dir, val_dir, test_dir]:
        for class_name in class_dirs:
            class_dir = os.path.join(split_dir, class_name.replace(' ', '_'))
            os.makedirs(class_dir, exist_ok=True)
    
    # Track counts
    class_counts = {class_name: 0 for class_name in class_dirs}
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    # For each class, find all label files and match with image files
    for class_name in class_dirs:
        class_label_dir = os.path.join(field_labels_dir, class_name)
        label_files = [f for f in os.listdir(class_label_dir) if f.endswith('.png')]
        
        # Count only files in this specific class directory
        class_label_count = len(label_files)
        print(f"Found {class_label_count} label files for class {class_name}")
        
        # Limit samples per class if specified
        if limit_per_class and class_label_count > limit_per_class:
            label_files = label_files[:limit_per_class]
            print(f"Limited to {limit_per_class} samples for class {class_name}")
        
        # Process each label file
        for label_file in label_files:
            # Get base filename without extension
            file_id = os.path.splitext(label_file)[0]
            
            # Find corresponding image file
            image_file = f"{file_id}.jpg"
            if not os.path.exists(os.path.join(field_images_dir, image_file)):
                image_file = f"{file_id}.png"
                if not os.path.exists(os.path.join(field_images_dir, image_file)):
                    print(f"Warning: Could not find image for label {label_file}")
                    continue
            
            # Get corresponding mask file
            mask_file = f"{file_id}.png"
            mask_path = os.path.join(field_masks_dir, mask_file)
            
            # Determine split based on random value
            rand_val = np.random.random()
            if rand_val < test_split:
                split = 'test'
            elif rand_val < test_split + val_split:
                split = 'val'
            else:
                split = 'train'
            
            # Create destination paths
            dest_class_dir = os.path.join(output_dir, split, class_name)
            dest_image_path = os.path.join(dest_class_dir, image_file)
            dest_label_path = os.path.join(dest_class_dir, label_file)
            
            # Copy files
            try:
                # Copy image
                image_path = os.path.join(field_images_dir, image_file)
                os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
                with open(image_path, 'rb') as src, open(dest_image_path, 'wb') as dst:
                    dst.write(src.read())
                
                # Copy label
                label_path = os.path.join(class_label_dir, label_file)
                os.makedirs(os.path.dirname(dest_label_path), exist_ok=True)
                with open(label_path, 'rb') as src, open(dest_label_path, 'wb') as dst:
                    dst.write(src.read())
                
                # Copy mask if it exists
                if os.path.exists(mask_path):
                    dest_mask_path = os.path.join(dest_class_dir, f"{file_id}_mask.png")
                    os.makedirs(os.path.dirname(dest_mask_path), exist_ok=True)
                    with open(mask_path, 'rb') as src, open(dest_mask_path, 'wb') as dst:
                        dst.write(src.read())
                
                # Update counts
                class_counts[class_name] += 1
                split_counts[split] += 1
                
                # Print progress periodically
                if (class_counts[class_name] % 100) == 1:
                    print(f"Processed {class_counts[class_name]} samples for class {class_name}")
                
            except Exception as e:
                print(f"Error processing {file_id}: {str(e)}")
    
    # Save dataset stats
    dataset_stats = {
        'total_samples': sum(split_counts.values()),
        'splits': split_counts,
        'class_distribution': class_counts
    }
    
    with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    
    print("\nDataset creation summary:")
    print(f"Total samples: {dataset_stats['total_samples']}")
    for split, count in split_counts.items():
        print(f"{split.capitalize()} set: {count} samples")
    
    return dataset_stats

def visualize_sample(image_path, label_path, mask_path=None, figsize=(15, 5)):
    """Visualize a sample image with its label and mask"""
    image = np.array(Image.open(image_path))
    label = np.array(Image.open(label_path))
    
    num_plots = 3 if mask_path else 2
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    # Display the image
    axes[0].imshow(image)
    axes[0].set_title('Field Image')
    axes[0].axis('off')
    
    # Display the label
    axes[1].imshow(label, cmap='gray')
    axes[1].set_title('Label')
    axes[1].axis('off')
    
    # Display the mask if available
    if mask_path:
        mask = np.array(Image.open(mask_path))
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title('Mask')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_label_colormap():
    """Create a colormap for visualizing multi-class labels"""
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]  # Background
    colormap[1] = [255, 0, 0]  # Class 1 (Red)
    colormap[2] = [0, 255, 0]  # Class 2 (Green)
    colormap[3] = [0, 0, 255]  # Class 3 (Blue)
    colormap[4] = [255, 255, 0]  # Class 4 (Yellow)
    colormap[5] = [255, 0, 255]  # Class 5 (Magenta)
    colormap[6] = [0, 255, 255]  # Class 6 (Cyan)
    colormap[7] = [128, 0, 0]  # Class 7 (Dark Red)
    colormap[8] = [0, 128, 0]  # Class 8 (Dark Green)
    colormap[9] = [0, 0, 128]  # Class 9 (Dark Blue)
    
    return colormap

def overlay_label_on_image(image, label, alpha=0.5):
    """Overlay a label on an image with transparency"""
    # Create a colormap for the labels
    colormap = create_label_colormap()
    
    # Apply the colormap to the label
    label_colored = colormap[label]
    
    # Convert image to float for blending
    image_float = image.astype(float)
    label_float = label_colored.astype(float)
    
    # Create a mask where label is non-zero
    mask = (label > 0).astype(float)
    mask_expanded = np.expand_dims(mask, axis=2)
    
    # Blend the image and label using the mask
    blended = (1 - alpha * mask_expanded) * image_float + alpha * mask_expanded * label_float
    
    # Clip the result to valid range and convert back to uint8
    return np.clip(blended, 0, 255).astype(np.uint8)

def augment_image(image, label, mask=None):
    """Apply data augmentation to an image and its corresponding label and mask"""
    # Define augmentation techniques (simple ones for now)
    height, width = image.shape[:2]
    
    # Random horizontal flip
    if random.random() > 0.5:
        image = np.fliplr(image)
        label = np.fliplr(label)
        if mask is not None:
            mask = np.fliplr(mask)
    
    # Random vertical flip
    if random.random() > 0.5:
        image = np.flipud(image)
        label = np.flipud(label)
        if mask is not None:
            mask = np.flipud(mask)
    
    # Random rotation (90, 180, 270 degrees)
    k = random.randint(0, 3)
    if k > 0:
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        if mask is not None:
            mask = np.rot90(mask, k)
    
    # Random brightness adjustment
    if random.random() > 0.5:
        brightness_factor = random.uniform(0.8, 1.2)
        image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
    
    return image, label, mask

def get_class_counts(stats):
    """Get class counts from stats dictionary"""
    if 'class_counts' in stats:
        return stats['class_counts']
    elif 'categories' in stats:
        return {cat['name']: cat['count'] for cat in stats['categories']}
    else:
        raise ValueError("Could not find class counts in stats")

def get_class_names(stats):
    """Get class names from stats dictionary"""
    if 'classes' in stats:
        return stats['classes']
    elif 'categories' in stats:
        return [cat['name'] for cat in stats['categories']]
    elif 'label_count' in stats:
        # Extract class names from label_count
        return list(stats['label_count'].keys())
    else:
        raise ValueError("Could not find class names in stats")

def get_image_size_stats(stats):
    """Get image size statistics from stats dictionary"""
    if 'image_size_stats' in stats:
        return stats['image_size_stats']
    else:
        return {"message": "Image size statistics not available"}

def load_image(image_path):
    """Load an image from path"""
    return cv2.imread(image_path)

def load_mask(mask_path):
    """Load a mask from path"""
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

def visualize_image_with_mask(image, mask, alpha=0.5):
    """Visualize an image with its mask overlay"""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(mask, alpha=alpha, cmap='jet')
    plt.title('Mask Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_dataset(dataset_dir, output_dir, val_split=0.15, test_split=0.15, limit_per_class=None):
    """
    Create a dataset from field images and organize it into train/val/test splits
    
    Args:
        dataset_dir: Base directory containing the dataset
        output_dir: Directory to save the processed dataset
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        limit_per_class: Maximum number of samples per class (None for all)
        
    Returns:
        A dictionary with train/val/test counts and class distribution
    """
    field_images_dir = os.path.join(dataset_dir, 'field_images', 'rgb')  # RGB images directory
    field_masks_dir = os.path.join(dataset_dir, 'field_masks')
    field_labels_dir = os.path.join(dataset_dir, 'field_labels')
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(field_images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    print(f"Found {len(image_files)} image files in {field_images_dir}")
    
    # Get all class directories in field_labels
    class_dirs = [d for d in os.listdir(field_labels_dir) if os.path.isdir(os.path.join(field_labels_dir, d))]
    print(f"Found {len(class_dirs)} class directories: {class_dirs}")
    
    # Create class directories in train/val/test
    for split_dir in [train_dir, val_dir, test_dir]:
        for class_name in class_dirs:
            class_dir = os.path.join(split_dir, class_name.replace(' ', '_'))
            os.makedirs(class_dir, exist_ok=True)
    
    # Track counts
    class_counts = {class_name: 0 for class_name in class_dirs}
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    # For each class, find all label files and match with image files
    for class_name in class_dirs:
        class_label_dir = os.path.join(field_labels_dir, class_name)
        label_files = [f for f in os.listdir(class_label_dir) if f.endswith('.png')]
        
        # Count only files in this specific class directory
        class_label_count = len(label_files)
        print(f"Found {class_label_count} label files for class {class_name}")
        
        # Limit samples per class if specified
        if limit_per_class and class_label_count > limit_per_class:
            label_files = label_files[:limit_per_class]
            print(f"Limited to {limit_per_class} samples for class {class_name}")
        
        # Process each label file
        for label_file in label_files:
            # Get base filename without extension
            file_id = os.path.splitext(label_file)[0]
            
            # Find corresponding image file
            image_file = f"{file_id}.jpg"
            if not os.path.exists(os.path.join(field_images_dir, image_file)):
                image_file = f"{file_id}.png"
                if not os.path.exists(os.path.join(field_images_dir, image_file)):
                    print(f"Warning: Could not find image for label {label_file}")
                    continue
            
            # Get corresponding mask file
            mask_file = f"{file_id}.png"
            mask_path = os.path.join(field_masks_dir, mask_file)
            
            # Determine split based on random value
            rand_val = np.random.random()
            if rand_val < test_split:
                split = 'test'
            elif rand_val < test_split + val_split:
                split = 'val'
            else:
                split = 'train'
            
            # Create destination paths
            dest_class_dir = os.path.join(output_dir, split, class_name)
            dest_image_path = os.path.join(dest_class_dir, image_file)
            dest_label_path = os.path.join(dest_class_dir, label_file)
            
            # Copy files
            try:
                # Copy image
                image_path = os.path.join(field_images_dir, image_file)
                os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
                with open(image_path, 'rb') as src, open(dest_image_path, 'wb') as dst:
                    dst.write(src.read())
                
                # Copy label
                label_path = os.path.join(class_label_dir, label_file)
                os.makedirs(os.path.dirname(dest_label_path), exist_ok=True)
                with open(label_path, 'rb') as src, open(dest_label_path, 'wb') as dst:
                    dst.write(src.read())
                
                # Copy mask if it exists
                if os.path.exists(mask_path):
                    dest_mask_path = os.path.join(dest_class_dir, f"{file_id}_mask.png")
                    os.makedirs(os.path.dirname(dest_mask_path), exist_ok=True)
                    with open(mask_path, 'rb') as src, open(dest_mask_path, 'wb') as dst:
                        dst.write(src.read())
                
                # Update counts
                class_counts[class_name] += 1
                split_counts[split] += 1
                
                # Print progress periodically
                if (class_counts[class_name] % 100) == 1:
                    print(f"Processed {class_counts[class_name]} samples for class {class_name}")
                
            except Exception as e:
                print(f"Error processing {file_id}: {str(e)}")
    
    # Save dataset stats
    dataset_stats = {
        'total_samples': sum(split_counts.values()),
        'splits': split_counts,
        'class_distribution': class_counts
    }
    
    with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    
    print("\nDataset creation summary:")
    print(f"Total samples: {dataset_stats['total_samples']}")
    for split, count in split_counts.items():
        print(f"{split.capitalize()} set: {count} samples")
    
    return dataset_stats 