# from google.colab import drive
import os
import matplotlib.pyplot as plt

# Base directory where all image data is stored (as provided by you)
IMAGE_DATA_DIR = '../Image Data'

# Function to count images in a folder and its subfolders
def count_images_in_subfolders(parent_folder_path):
    # Define image extensions to check
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    classification_counts = {}
    folder_name_mapping = {}

    # Walk through the parent folder and its subfolders
    for root, dirs, files in os.walk(parent_folder_path):
        # Count image files in the current subfolder
        image_count = sum(1 for file in files if any(file.lower().endswith(ext) for ext in image_extensions))

        # If the current folder has image files, prompt for renaming
        if image_count > 0:
            original_name = os.path.basename(root)  # Use folder name as classification
            print(f"Found folder: '{original_name}' with {image_count} images.")
            new_name = input(f"Enter a new name for this folder (or type 'REMOVE' to exclude it): ").strip()
            if new_name.upper() == "REMOVE":
                print(f"Folder '{original_name}' will be excluded from the bar graph.")
                continue  # Skip this folder
            if new_name == "":
                new_name = original_name  # Keep the original name if no input is provided
            folder_name_mapping[original_name] = new_name
            classification_counts[new_name] = image_count

    return classification_counts, folder_name_mapping

# Specify the parent folder path (modify this to your folder path)
parent_folder_path = IMAGE_DATA_DIR

# Get image counts by classification and folder name mapping
classification_counts, folder_name_mapping = count_images_in_subfolders(parent_folder_path)

# Print the folder name mapping
print("\nFolder Name Mapping:")
for original, new in folder_name_mapping.items():
    print(f"'{original}' -> '{new}'")

# Plot the bar graph for subfolders with images
plt.figure(figsize=(12, 6))
bars = plt.bar(classification_counts.keys(), classification_counts.values(), color='skyblue')

# Add counts on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, str(height), ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45, ha='right')
plt.xlabel('Classifications')
plt.ylabel('Count of Images')
plt.title('Image Counts by Classification')
plt.tight_layout()
plt.show()