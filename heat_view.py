import jsonpickle
import matplotlib.pyplot as plt
import pandas as pd

JSON_FILE_PATH = r"saved_models/model1-UNet/"

FILES = [f"setting_{idx}" for idx in range(1,8)]

predictions = []

for file in FILES:

    with open(JSON_FILE_PATH + file + ".json", "r") as reader:
        data:dict = jsonpickle.decode(reader.read()) # type: ignore

    predictions.append(list(data["disclosed_predictions"].values()))

heat_df = pd.DataFrame(predictions, index=FILES, columns=list(data["disclosed_predictions"].keys()))

print(heat_df)

fig = plt.figure(1, (12,12))
plt.matshow(heat_df, 1)
plt.xticks(range(0,len(heat_df.columns)), list(heat_df.columns), rotation=45, ha="left")
plt.yticks(range(0,len(heat_df.index)),list(heat_df.index))
plt.colorbar(location="bottom")
fig.savefig("heatplot.png")