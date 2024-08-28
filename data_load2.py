import os
import numpy as np
from PIL import Image

def load_data(noisy_folder_path, clean_folder_path):

    def collect_images_from_subfolders(folder_path, exclude_folder):
        images = []
        for root, dirs, files in os.walk(folder_path):
            # Skip the clean folder to avoid collecting clean images
            if exclude_folder not in root:
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        images.append(os.path.join(root, file))
        return images

    # Collect all noisy files, excluding the clean folder
    noisy_files = collect_images_from_subfolders(noisy_folder_path, clean_folder_path)
    
    # Determine the number of noisy files and the shape of the images
    num_files = len(noisy_files)
    if num_files == 0:
        raise ValueError("No images found in the noisy folder or its subfolders.")

    # Get the shape of the first noisy image to initialize numpy arrays
    img_shape = np.array(Image.open(noisy_files[0])).shape
    noisy_data = np.zeros((num_files, img_shape[0], img_shape[1]), dtype=np.uint8)
    clean_data = np.zeros((num_files, img_shape[0], img_shape[1]), dtype=np.uint8)

    # Load the images into numpy arrays
    index = 0
    for noisy_file in noisy_files:
        # Extract the base filename to find the corresponding clean file
        base_filename = os.path.basename(noisy_file)
        clean_file = os.path.join(clean_folder_path, base_filename)

        if os.path.exists(clean_file):
            noisy_image = Image.open(noisy_file)
            clean_image = Image.open(clean_file)

            noisy_array = np.array(noisy_image)
            clean_array = np.array(clean_image)

            noisy_data[index] = noisy_array
            clean_data[index] = clean_array

            index += 1

    # Resize arrays to the actual number of valid images
    noisy_data = noisy_data[:index]
    clean_data = clean_data[:index]

    return noisy_data, clean_data
