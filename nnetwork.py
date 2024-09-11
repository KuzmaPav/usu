from time import sleep
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')

import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import jsonpickle
import keras
import gc

import data_load2 as data_load
#import model1 as model_lib
from losses_and_metrics import *
import model2 as model_lib

# Data 
DEFAULT_NOISE_DATA_PATH = r"data"
DEFAULT_CLEAN_DATA_PATH = r"data/original"

# Saving Folder
DEFAULT_SAVE_FOLDER = r"saved_models"

MODEL_TYPE = r"model2-Custom"

TEMP_KERAS = f"temp{str(np.random.random())[2:]}.keras"



def prepare_data(noise_data_path:str, clean_data_path:str):
    """ Specifically tailored function to collect data for this training

    Returns: tuple of X data and Y data with channels
    """ 

    x_data, y_data = data_load.load_data(noise_data_path, clean_data_path)
    x_data = np.expand_dims(x_data, -1)
    y_data = np.expand_dims(y_data, -1)
    return x_data, y_data







def setup_model(x_train:np.ndarray, y_train:np.ndarray, model_kwargs:dict, fit_kwargs:dict) -> keras.Model:
    """ Function that setups model and trains it on provided data

    Returns: trained model and dictionary of model statistics
    """
    
    model = model_lib.get_model(**model_kwargs)
    model.summary()
    
    model.fit(x_train, y_train, **fit_kwargs)
    model.save(TEMP_KERAS)

    return model


def evaluate_model(model, x_test, y_test):
    evaluation_values = model.evaluate(x_test, y_test)
    
    evaluation_dict = {"loss": evaluation_values[0]}
    evaluation_dict.update({metric.name: value for metric, value in zip(model.metrics[1].metrics, evaluation_values[1:], strict=True)}) # type: ignore
    
    return evaluation_dict


def predict(model:keras.Model, noise_data:np.ndarray, clean_data:np.ndarray|None=None, plot:bool=False) -> np.ndarray:
    """ Function that predicts provided data in provided model

    Returns: predicted data
    """

    noise_data = np.expand_dims(noise_data, axis=0)
    predicted_data = model.predict(noise_data)
    if plot:
        plot_data(noise_data, predicted_data, clean_data)
    return predicted_data




def plot_data(noise_data:np.ndarray, nn_cleansed_data:np.ndarray, clean_data:np.ndarray|None=None, show:bool=True): # -> plt.Figure
    """ Function that prepares figure with X data, predicted data and Y data if provided

    plots it if parameter provided 

    Returns: plot figure
    """

    # Checks if data have wrong shape and readjusts it (model uses batches, plot don't know what that is)
    noise_data = np.squeeze(noise_data, axis=0) if len(noise_data.shape) == 4 else noise_data
    nn_cleansed_data = np.squeeze(nn_cleansed_data, axis=0) if len(nn_cleansed_data.shape) == 4 else nn_cleansed_data

    if clean_data is None:
        cols = 2
    else:
        cols = 3
        clean_data = np.squeeze(clean_data, axis=0) if len(clean_data.shape) == 4 else clean_data


    # Turns off axis lines and values for plots
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = False
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.labelleft'] = False

    figure = plt.figure("Result_show")
    
    plt.subplot(1, cols, 1)
    plt.imshow(noise_data)
    plt.title("Noised image")

    plt.subplot(1, cols, 2)
    plt.imshow(nn_cleansed_data)
    plt.title("NN Cleansed image")

    if cols == 3:
        plt.subplot(1, 3, 3)
        plt.imshow(clean_data) # type: ignore
        plt.title("Clean image")

    if show:
        plt.show()

    return figure




def save_model_info(model_settings:dict, fit_settings:dict, model_evaluation:dict, plot_figure, keras_model:keras.Model|None=None):
    """ Function that takes various parameters and saves them into json file

    function finds the last file in save folder and based on that saves the information in files with incremented number

    Returns: None
    """
    model_dir = os.path.join(DEFAULT_SAVE_FOLDER, MODEL_TYPE)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    existing_files = [f for f in os.listdir(model_dir) if f.startswith("set_") and f.endswith(".json")]
    if existing_files:
        last_number = max([int(f.split('_')[1].split('.')[0]) for f in existing_files])
        next_number = last_number + 1
    else:
        next_number = 1

    settings_name = os.path.join(model_dir, f"setting_{next_number}")

    with open(settings_name + ".json", "w") as writer:
        writer.write(str(jsonpickle.encode({
            "model_settings": model_settings,
            "fit_settings": fit_settings,
            "model_evaluation": model_evaluation,
        }, indent=4)))

    plot_figure.savefig(settings_name)

    if keras_model is not None:
        keras_model.save(settings_name + ".keras")



def main(model_settings, fit_settings, noise_data_path=DEFAULT_NOISE_DATA_PATH, clean_data_path=DEFAULT_CLEAN_DATA_PATH, skip_user = False):

    x_data, y_data = prepare_data(noise_data_path, clean_data_path)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


    model = setup_model(x_train, y_train, {"input_shape": x_data.shape[1:], **model_settings}, fit_settings)
    

    model_evaluation = evaluate_model(model, x_test, y_test)

    random_idx = np.random.randint(x_data.shape[0])
    random_noise_image = x_data[random_idx]
    clean_image = y_data[random_idx]

    predicted_image = predict(model, random_noise_image, clean_image, False)

    plot_figure = plot_data(random_noise_image, predicted_image, clean_image)


    if skip_user == True:
        save_keras = True

    else:
        while True:
            user_input = input("Do you wish to save the keras model? [Y/N] ")

            if user_input.lower() == "y":
                save_keras = True
                break

            elif user_input.lower() == "n":
                break

    save_model_info(model_settings, fit_settings, model_evaluation, plot_figure, model if save_keras else None)

    os.remove(TEMP_KERAS)


if __name__ == "__main__":

    #model1
    model_settings = {
        "depth": 5,
        "initial_filters": 32,
        "optimizer" : "adam",
        "loss_fce" : ssim_loss,
        "metrics": [psnr_metric, "accuracy"],
    }

    #model2
    model_settings = {
        "paths" : 3, 
        "filters" : 256, 
        "classification_count" : 10,
        "class_size" : 16,
        "optimizer" : "adam",
        "loss_fce" : ssim_loss,
        "metrics": [psnr_metric, "accuracy"],
    }


    fit_settings = {
        "epochs": 1,
        "batch_size": 1,
        "validation_split": 0.2,
        "steps_per_epoch": 100,
    }

    main(model_settings, fit_settings, skip_user=True)

