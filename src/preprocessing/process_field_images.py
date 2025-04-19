import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import utils
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.field_image_utils import (
    get_field_image_paths,
    load_field_stats,
    get_field_class_names,
    get_image_label_pairs,
    create_dataset,
    visualize_sample
)
from utils.paths import ensure_dir_exists

def explore_dataset(base_dir):
    """Explore the dataset and print statistics"""
    # Load field stats
    stats = load_field_stats(base_dir)
    
    print("\n===== Field Image Dataset Exploration =====")
    print(f"Total number of images: {stats['num_images']}")
    
    # Print label distribution
    print("\nLabel distribution:")
    for label, count in stats['label_counts'].items():
        print(f"  {label}: {count} images")
    
    # Get class names from directory structure
    class_names = get_field_class_names(base_dir)
    print(f"\nFound {len(class_names)} classes: {', '.join(class_names)}")
    
    # Print image size statistics
    print("\nImage size statistics:")
    for size, count in stats['image_size_counts'].items():
        print(f"  {size}: {count} images")
    
    return stats, class_names

def analyze_class_samples(base_dir, class_names, sample_limit=5):
    """Analyze samples from each class"""
    print("\n===== Sample Analysis =====")
    
    for class_name in class_names:
        print(f"\nAnalyzing class: {class_name}")
        
        # Get image-label pairs for this class
        pairs = get_image_label_pairs(base_dir, class_name, limit=sample_limit)
        
        if not pairs:
            print(f"  No samples found for class {class_name}")
            continue
        
        print(f"  Found {len(pairs)} samples")
        
        # Show sample paths
        sample_pair = pairs[0]
        print(f"  Sample image path: {sample_pair['image_path']}")
        print(f"  Sample label path: {sample_pair['label_path']}")
        if sample_pair['mask_path']:
            print(f"  Sample mask path: {sample_pair['mask_path']}")
        
        # Visualize the first sample
        print(f"  Visualizing first sample for class {class_name}...")
        _ = visualize_sample(
            sample_pair['image_path'],
            sample_pair['label_path'],
            sample_pair['mask_path']
        )

def main():
    parser = argparse.ArgumentParser(description='Process field image dataset')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory of the field image dataset')
    parser.add_argument('--output_dir', type=str, default='data/processed/field_images',
                        help='Output directory for processed data')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Proportion of data to use for validation')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Proportion of data to use for testing')
    parser.add_argument('--limit_per_class', type=int, default=None,
                        help='Limit the number of samples per class (for testing)')
    parser.add_argument('--explore_only', action='store_true',
                        help='Only explore the dataset, do not process it')
    
    args = parser.parse_args()
    
    # Explore the dataset
    stats, class_names = explore_dataset(args.base_dir)
    
    if args.explore_only:
        # Analyze samples from each class
        analyze_class_samples(args.base_dir, class_names)
        return
    
    # Process the dataset
    print(f"\nProcessing dataset and saving to {args.output_dir}...")
    ensure_dir_exists(args.output_dir)
    
    # Create the dataset
    dataset_info = create_dataset(
        args.base_dir,
        args.output_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        limit_per_class=args.limit_per_class
    )
    
    print("\nDataset created successfully!")
    print(f"Total samples: {dataset_info['samples']['total']}")
    print(f"  - Training: {dataset_info['samples']['train']}")
    print(f"  - Validation: {dataset_info['samples']['val']}")
    print(f"  - Testing: {dataset_info['samples']['test']}")
    
    print("\nClass distribution:")
    for class_name, count in dataset_info['class_distribution'].items():
        print(f"  {class_name}: {count} samples")

if __name__ == '__main__':
    main() 