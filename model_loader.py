import keras
import nnetwork
import numpy as np
import matplotlib.pyplot as plt


#MODEL_PATH = r"saved_models/model1-UNet/setting_7.keras"
MODEL_PATH = r"saved_models/model2-Custom/temp060720674830973875.keras"

x_data, y_data = nnetwork.prepare_data(nnetwork.DEFAULT_NOISE_DATA_PATH, nnetwork.DEFAULT_CLEAN_DATA_PATH)

model:keras.Model = keras.models.load_model(MODEL_PATH) # type: ignore

random_idx = np.random.randint(x_data.shape[0])
random_noise_image = x_data[random_idx]
clean_image = y_data[random_idx]

idx = 901
random_noise_image = x_data[idx]
clean_image = y_data[idx]



# WSL linux on windows have no graphical interface, therefore it cannot show plot -> must save it and view it differently
predicted_image = nnetwork.predict(model, random_noise_image, clean_image, False) #if can show plots, False -> True and skip further code

figure = nnetwork.plot_data(random_noise_image, predicted_image, clean_image)
figure.savefig("currentplot")