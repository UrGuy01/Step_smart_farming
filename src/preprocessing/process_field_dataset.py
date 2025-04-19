import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to system path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.paths import ensure_dir_exists, get_project_root, get_data_dir, get_processed_data_dir
from utils.field_image_utils import (
    load_field_stats, 
    get_class_counts, 
    get_class_names,
    create_dataset,
    visualize_sample
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process field image dataset")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=None,
        help="Directory containing the field image dataset (if not specified, will search in project data directory)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save processed dataset (default: processed_data/field_dataset)"
    )
    parser.add_argument(
        "--val_split", 
        type=float, 
        default=0.15,
        help="Fraction of data to use for validation (default: 0.15)"
    )
    parser.add_argument(
        "--test_split", 
        type=float, 
        default=0.15,
        help="Fraction of data to use for testing (default: 0.15)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit samples per class (default: None, use all available)"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Visualize dataset statistics after processing"
    )
    
    return parser.parse_args()

def find_dataset_dir(base_dir=None):
    """Find the dataset directory by looking for common subdirectories"""
    if base_dir is None:
        base_dir = get_data_dir()
    
    # Look for directories that match the expected dataset structure
    potential_dirs = []
    
    for root, dirs, files in os.walk(base_dir):
        # Check if this directory has the expected subdirectories
        if ('field_images' in dirs and 
            'field_masks' in dirs and 
            'field_labels' in dirs):
            return root
        
        # Check if we found the directory named data2017_miniscale
        if os.path.basename(root) == 'data2017_miniscale':
            return root
        
        # Add directories that contain 'field' in their name to potential dirs
        for d in dirs:
            if 'field' in d.lower():
                potential_dirs.append(os.path.join(root, d))
    
    # If we didn't find an exact match, check the potential directories
    for d in potential_dirs:
        if os.path.exists(os.path.join(d, 'field_images')):
            return d
    
    # If we still didn't find anything, look for the tar.gz file
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.tar.gz') and 'data' in file.lower():
                print(f"Found dataset archive: {os.path.join(root, file)}")
                print("Please extract it first and run the script again.")
                sys.exit(1)
    
    raise FileNotFoundError("Could not find the field dataset directory. Please specify with --data_dir")

def visualize_dataset_stats(dataset_info):
    """Visualize dataset statistics"""
    print("\nDataset Statistics:")
    print(f"Total samples: {dataset_info['samples']['total']}")
    print(f"Training samples: {dataset_info['samples']['train']}")
    print(f"Validation samples: {dataset_info['samples']['val']}")
    print(f"Test samples: {dataset_info['samples']['test']}")
    
    # Visualize class distribution
    class_names = list(dataset_info['class_distribution'].keys())
    class_counts = list(dataset_info['class_distribution'].values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to process the field image dataset"""
    args = parse_args()
    
    # Find dataset directory if not specified
    if args.data_dir is None:
        try:
            args.data_dir = find_dataset_dir()
            print(f"Found dataset directory: {args.data_dir}")
        except FileNotFoundError as e:
            print(str(e))
            return
    
    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(get_processed_data_dir(), 'field_dataset')
    
    # Ensure output directory exists
    ensure_dir_exists(args.output_dir)
    
    print(f"\nProcessing field image dataset:")
    print(f"Input directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Validation split: {args.val_split}")
    print(f"Test split: {args.test_split}")
    print(f"Samples per class limit: {args.limit if args.limit else 'All'}")
    
    # Print dataset structure for debugging
    print("\nDataset Structure:")
    print(f"Field images directory exists: {os.path.exists(os.path.join(args.data_dir, 'field_images'))}")
    print(f"Field labels directory exists: {os.path.exists(os.path.join(args.data_dir, 'field_labels'))}")
    print(f"Field masks directory exists: {os.path.exists(os.path.join(args.data_dir, 'field_masks'))}")
    
    # Check field_images subdirectories
    field_images_dir = os.path.join(args.data_dir, 'field_images')
    if os.path.exists(field_images_dir):
        print(f"Field images subdirectories: {os.listdir(field_images_dir)}")
    
    # Check field_labels subdirectories
    field_labels_dir = os.path.join(args.data_dir, 'field_labels')
    if os.path.exists(field_labels_dir):
        print(f"Field labels subdirectories: {os.listdir(field_labels_dir)}")
    
    # Process the dataset
    try:
        dataset_info = create_dataset(
            args.data_dir,
            args.output_dir,
            val_split=args.val_split,
            test_split=args.test_split,
            limit_per_class=args.limit
        )
        
        print("\nDataset processing completed successfully!")
        
        # Visualize dataset statistics if requested
        if args.visualize:
            visualize_dataset_stats(dataset_info)
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nProcessed dataset saved to: {args.output_dir}")
    print("Done!")

if __name__ == "__main__":
    main() 