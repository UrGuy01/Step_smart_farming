import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.field_image_utils import (
    visualize_sample,
    load_image,
    load_mask,
    overlay_label_on_image,
    create_label_colormap
)

def load_dataset_stats(dataset_dir):
    """Load dataset statistics from JSON file"""
    stats_file = os.path.join(dataset_dir, 'dataset_stats.json')
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    return stats

def plot_class_distribution(stats):
    """Plot class distribution as a bar chart"""
    class_names = list(stats['class_distribution'].keys())
    class_counts = list(stats['class_distribution'].values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_counts)
    
    # Add count labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{int(height)}', ha='center', va='bottom')
    
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/visualizations/class_distribution.png')
    plt.show()

def plot_dataset_split(stats):
    """Plot dataset split as a pie chart"""
    split_names = list(stats['splits'].keys())
    split_counts = list(stats['splits'].values())
    
    plt.figure(figsize=(8, 8))
    plt.pie(split_counts, labels=split_names, autopct='%1.1f%%', startangle=90)
    plt.title('Dataset Split')
    plt.axis('equal')
    plt.savefig('results/visualizations/dataset_split.png')
    plt.show()

def visualize_random_samples(dataset_dir, num_samples=3, split='train'):
    """Visualize random samples from the dataset"""
    split_dir = os.path.join(dataset_dir, split)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    
    plt.figure(figsize=(15, num_samples*5))
    
    for i in range(num_samples):
        # Randomly select a class
        class_name = random.choice(class_dirs)
        class_dir = os.path.join(split_dir, class_name)
        
        # Get all image files in this class directory
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        
        if not image_files:
            continue
        
        # Randomly select an image
        image_file = random.choice(image_files)
        image_path = os.path.join(class_dir, image_file)
        
        # Get corresponding label file
        label_file = os.path.splitext(image_file)[0] + '.png'
        label_path = os.path.join(class_dir, label_file)
        
        # Check if mask exists
        mask_file = os.path.splitext(image_file)[0] + '_mask.png'
        mask_path = os.path.join(class_dir, mask_file)
        
        if not os.path.exists(mask_path):
            mask_path = None
        
        # Load the image and label
        image = np.array(Image.open(image_path))
        label = np.array(Image.open(label_path))
        
        # Plot the images
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(image)
        plt.title(f'Image - Class: {class_name}')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(label, cmap='gray')
        plt.title('Label')
        plt.axis('off')
        
        # If mask exists, show it as the third image
        if mask_path:
            mask = np.array(Image.open(mask_path))
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.imshow(mask, cmap='gray')
            plt.title('Mask')
            plt.axis('off')
        else:
            # If no mask, show overlayed image instead
            overlayed = overlay_label_on_image(image, label)
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.imshow(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
            plt.title('Overlay')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/sample_images.png')
    plt.show()

def analyze_image_characteristics(dataset_dir, split='train', sample_size=100):
    """Analyze image characteristics like dimensions, channels, etc."""
    split_dir = os.path.join(dataset_dir, split)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    
    # Store image characteristics
    image_dimensions = []
    image_channels = []
    label_unique_values = []
    
    # Randomly sample images
    samples_collected = 0
    
    while samples_collected < sample_size:
        # Randomly select a class
        class_name = random.choice(class_dirs)
        class_dir = os.path.join(split_dir, class_name)
        
        # Get all image files in this class directory
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        
        if not image_files:
            continue
        
        # Randomly select an image
        image_file = random.choice(image_files)
        image_path = os.path.join(class_dir, image_file)
        
        # Get corresponding label file
        label_file = os.path.splitext(image_file)[0] + '.png'
        label_path = os.path.join(class_dir, label_file)
        
        # Load the image and label
        image = np.array(Image.open(image_path))
        label = np.array(Image.open(label_path))
        
        # Record dimensions
        image_dimensions.append(image.shape[:2])
        
        # Record number of channels
        if len(image.shape) > 2:
            image_channels.append(image.shape[2])
        else:
            image_channels.append(1)
        
        # Record unique values in label
        label_unique_values.append(np.unique(label))
        
        samples_collected += 1
    
    # Print summary statistics
    print("\nImage Characteristics Summary:")
    print(f"Sample size: {sample_size} images")
    
    # Dimensions
    height_values = [dim[0] for dim in image_dimensions]
    width_values = [dim[1] for dim in image_dimensions]
    print(f"Image dimensions:")
    print(f"  - Height: min={min(height_values)}, max={max(height_values)}, mean={np.mean(height_values):.1f}")
    print(f"  - Width: min={min(width_values)}, max={max(width_values)}, mean={np.mean(width_values):.1f}")
    
    # Channels
    print(f"Image channels: {set(image_channels)}")
    
    # Label values
    all_label_values = set()
    for values in label_unique_values:
        all_label_values.update(values)
    print(f"Label unique values: {sorted(all_label_values)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Explore field image dataset')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory containing the processed dataset')
    parser.add_argument('--visualize_samples', action='store_true',
                        help='Visualize random samples from the dataset')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of random samples to visualize')
    parser.add_argument('--output_file', type=str, default='results/visualizations/dataset_analysis.txt',
                        help='File to save the analysis results')
    
    args = parser.parse_args()
    
    # Create visualization directory if it doesn't exist
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Redirect output to file
    original_stdout = sys.stdout
    with open(args.output_file, 'w') as f:
        sys.stdout = f
        
        # Load dataset statistics
        stats = load_dataset_stats(args.dataset_dir)
        
        print("\n===== Dataset Statistics =====")
        print(f"Total samples: {stats['total_samples']}")
        
        print("\nSplit distribution:")
        for split, count in stats['splits'].items():
            print(f"  {split}: {count} samples ({count/stats['total_samples']*100:.1f}%)")
        
        print("\nClass distribution:")
        for class_name, count in stats['class_distribution'].items():
            print(f"  {class_name}: {count} samples ({count/stats['total_samples']*100:.1f}%)")
        
        # Analyze image characteristics without showing plots
        analyze_image_characteristics(args.dataset_dir)
    
    # Reset stdout
    sys.stdout = original_stdout
    
    print(f"Analysis results saved to {args.output_file}")
    
    # Generate plots
    plot_class_distribution(stats)
    plot_dataset_split(stats)
    
    # Visualize samples if requested
    if args.visualize_samples:
        visualize_random_samples(args.dataset_dir, num_samples=args.num_samples)
    
    print("Exploration complete! Visualization images saved to results/visualizations/")

if __name__ == "__main__":
    main() 