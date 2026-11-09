import os
import pandas as pd

# Base directory where all image data is stored (as provided by you)
IMAGE_DATA_DIR = "../Image Data"

def resolve_dataset_dir(base_dir: str) -> str:
    """
    Resolve the dataset root directory under the base image directory.
    Your folder may contain `PlantVillage/` or `PlantVillage/PlantVillage/`.
    """
    candidates = [
        os.path.join(base_dir, "PlantVillage", "PlantVillage"),
        os.path.join(base_dir, "PlantVillage"),
        base_dir,
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"Could not find a dataset directory under {base_dir}. Checked: {candidates}")

def create_metadata_from_parent_directory(parent_directory, output_csv):
    """
    Automatically discover subdirectories in the parent directory and create metadata.
    Each subdirectory name becomes the 'dx' (diagnosis/label) category.
    """
    metadata = []  # List to store metadata rows
    
    if not os.path.isdir(parent_directory):
        print(f"Parent directory '{parent_directory}' does not exist.")
        return
    
    # Get all subdirectories in the parent directory
    subdirectories = [d for d in os.listdir(parent_directory) 
                      if os.path.isdir(os.path.join(parent_directory, d))]
    
    print(f"Found {len(subdirectories)} subdirectories:")
    for subdir in subdirectories:
        print(f"  - {subdir}")
    
    # Process each subdirectory
    for subdir in subdirectories:
        subdir_path = os.path.join(parent_directory, subdir)
        folder_label = subdir  # Use subdirectory name as the label
        
        # Get all files in the subdirectory
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)
            if os.path.isfile(file_path):  # Only process files
                metadata.append({"image_id": file, "dx": folder_label})
    
    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(metadata)
    df.to_csv(output_csv, index=False)
    print(f"\nMetadata saved to {output_csv}")
    print(f"Total images processed: {len(metadata)}")

# Usage
parent_directory = resolve_dataset_dir(IMAGE_DATA_DIR)
output_csv = os.path.join(IMAGE_DATA_DIR, "plant_health_metadata.csv")
create_metadata_from_parent_directory(parent_directory, output_csv)
