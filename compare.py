import matplotlib.pyplot as plt
import data_load2 as data_load
import numpy as np

CLEAN_PATH = r"data/original"
DATA_PATH = r"data"

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


# Turns off axis lines and values for plots
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False


x_data, y_data = data_load.load_data(DATA_PATH, CLEAN_PATH)

plot_figure = plt.figure("Noise comparison", (12, 12*5))


for loop_idx, d in enumerate(list(data_splitter.items())):

    key, value = d

    random_idx = np.random.randint(value[0], value[1])

    plt.subplot(len(data_splitter.keys()), 2, (loop_idx * 2) + 1)
    plt.title(key)
    plt.imshow(x_data[random_idx])
    plt.subplot(len(data_splitter.keys()), 2, (loop_idx *2) + 2)
    plt.imshow(y_data[random_idx])

plot_figure.savefig("comparison")