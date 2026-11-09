import os
import random
from PIL import Image

# Base directory where all image data is stored (as provided by you)
IMAGE_DATA_DIR = "../Image Data"

# Function to rotate an image by a random degree and save it
def rotate_and_save_image(image_path, output_path, degrees):
    with Image.open(image_path) as img:
        rotated_image = img.rotate(degrees, expand=True)
        filename, ext = os.path.splitext(os.path.basename(image_path))
        new_filename = f"{filename}_{degrees}{ext}"
        new_image_path = os.path.join(output_path, new_filename)
        rotated_image.save(new_image_path)
        print(f"Saved {new_filename} into {output_path}")

# Function to process specific subdirectories and generate 5x images
def process_specific_subdirectories(input_directory, output_directory, subdirectories):
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(input_directory, subdirectory)
        if os.path.isdir(subdirectory_path):
            for root, _, files in os.walk(subdirectory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, input_directory)
                    output_path = os.path.join(output_directory, relative_path)
                    os.makedirs(output_path, exist_ok=True)
                    
                    # Calculate how many rotations needed per image to get 5x output
                    rotations_needed = 5
                    for _ in range(rotations_needed):
                        degrees = random.randint(0, 360)
                        rotate_and_save_image(file_path, output_path, degrees)
        else:
            print(f"Subdirectory '{subdirectory}' does not exist in '{input_directory}'")

# Example usage
input_directory = IMAGE_DATA_DIR
output_directory = os.path.join(IMAGE_DATA_DIR, "rotated_output")
subdirectories = ['FINALbasalcell', 'FINALdermatofibroma', 'FINALvascularlesion']  # List the subdirectories you want to process
process_specific_subdirectories(input_directory, output_directory, subdirectories)
