import os
import jsonpickle
import numpy as np
import tensorflow as tf
import keras
import data_load2 as data_load
from losses_and_metrics import *

# Paths
CLEAN_PATH = r"data/original"
DATA_PATH = r"data"
MODEL_PATH = r"saved_models/model1-UNet/setting_7" 


# Data Splitter
data_splitter = {
    "poisson_intensity_0.1": (600, 699),
    "poisson_intensity_0.2": (700, 799),
    "poisson_intensity_0.3": (800, 899),
    "salt_and_pepper_speckle_intensity_0.01_0.1": (1100, 1199),
    "salt_and_pepper_speckle_intensity_0.03_0.3": (1000, 1099),
    "salt_and_pepper_speckle_intensity_0.05_0.2": (900, 999),
    "salt_pepper_intensity_0.01": (300, 399),
    "salt_pepper_intensity_0.03": (400, 499),
    "salt_pepper_intensity_0.05": (500, 599),
    "speckle_intensity_0.1": (100, 199),
    "speckle_intensity_0.05": (0, 99),
    "speckle_intensity_0.15": (200, 299),
}



def prepare_data(noise_data_path:str, clean_data_path:str):
    """ Specifically tailored function to collect data for this training

    Returns: tuple of X data and Y data with channels
    """ 

    x_data, y_data = data_load.load_data(noise_data_path, clean_data_path)
    x_data = np.expand_dims(x_data, -1)
    y_data = np.expand_dims(y_data, -1)
    return x_data, y_data



# Function to calculate similarity
def calculate_similarity(original, predicted):
    original = original.squeeze()  # Remove batch dimension
    predicted = predicted.squeeze()
    
    # Calculate SSIM
    ssim_value = ssim_metric(original, predicted)

    return ssim_value

# Update JSON with predictions and similarity metrics
def update_json(json_path, data_splitter, model:keras.Model, clean_path, data_path):
    with open(json_path, 'r') as json_file:
        json_data:dict = jsonpickle.decode(json_file.read()) # type: ignore
    
    x_data, y_data = prepare_data(data_path, clean_path)
    
    json_data["disclosed_predictions"] = dict()

    # Iterate over data_splitter
    for key, (start_idx, end_idx) in data_splitter.items(): 
        predicted_imgs = model.predict(x_data[start_idx : end_idx], 32)

        predictions = tf.image.ssim(tf.cast(y_data[start_idx : end_idx], tf.float32), predicted_imgs, max_val=255.0)

            
        # Update the JSON data
        json_data["disclosed_predictions"][key] = float(np.average(predictions)) # type: ignore


    # Write updated data to JSON
    with open(json_path, 'w') as json_file:
        json_file.write(jsonpickle.encode(json_data, indent=4))  # type: ignore
    

model:keras.Model = keras.models.load_model(MODEL_PATH + ".keras") # type: ignore

# Run the update
update_json(MODEL_PATH + ".json", data_splitter, model, CLEAN_PATH, DATA_PATH)
