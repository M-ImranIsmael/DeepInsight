#%% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard

#%% Step 1) Data Loading
CSV_PATH = os.path.join(os.getcwd(), "datasets", "cases_malaysia_train.csv")
df = pd.read_csv(CSV_PATH)

#%% Step 2) EDA
print(df.head(3))
print(df.info())
print(df.shape)

#%% Step 3) Data Cleaning
# Format date column using datetime
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
# Make sure to order by ascending date
df = df.sort_values("date")

print(df.head(20))

#%%
# Convert cases_new to numeric data type
df["cases_new"] = pd.to_numeric(df["cases_new"], errors="coerce")

# Interpolation to Handle missing values
df["cases_new"] = df["cases_new"].interpolate(method="polynomial", order=2)

print(df.info())
print(df.tail(5))
#%%

# Visual EDA: Daily New COVID-19 Cases in Malaysia (Train datasets)
plt.figure()
plt.plot(df["cases_new"], color="maroon")
plt.xlabel("Days")
plt.ylabel("New Cases")
plt.title("Daily New COVID-19 Cases in Malaysia")
plt.savefig("pictures/Imran_new_cases_plot.png", dpi=300, bbox_inches="tight")
plt.show()

#%% Step 4) Feature Selections

data = df["cases_new"].values  # To obtain data in ndarray format

#%% Step 5) Data Preprocessing

mms = MinMaxScaler()
data = mms.fit_transform(np.expand_dims(data, axis=-1))

#%%
X_train, y_train = [], []  # Instantiate empty list

win_size = 30

for i in range(win_size, len(data)):
    X_train.append(data[i - win_size : i])
    y_train.append(data[i])

# Convert list into ndarray
X_train = np.array(X_train)
y_train = np.array(y_train)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.3, random_state=123, shuffle=True
)


#%%
print(np.isnan(X_train).any())
print(np.isnan(y_train).any())
#%% Step 6) Model Development

input_shape = np.shape(X_train)[1:]
model = Sequential()
model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation="linear"))
model.summary()
plot_model(model, show_shapes=True, to_file="pictures/Imran_model_architecture.png")

optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss="mse", metrics=["mape", "mse"])

#%% Tensorboard and fitting

parent_folder = "tensorboard_log"
log_dir = os.path.join(
    os.getcwd(), parent_folder, datetime.datetime.now().strftime("%Y&m%d-%H%M%S")
)
tb_callback = TensorBoard(log_dir=log_dir)

# To include early stopping to prevent overfitting

es_callback = EarlyStopping(
    monitor="val_mape", mode="min", patience=90, restore_best_weights=True
)

hist = model.fit(
    X_train,
    y_train,
    validation_data=[X_test, y_test],
    epochs=300,
    callbacks=[tb_callback, es_callback],
)

#%% Step 7) Model Analysis

print(hist.history.keys())
plt.figure()
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.legend(["Training Loss", "Validation Loss"])
plt.savefig("pictures/Imran_trainval_loss_.png", dpi=300, bbox_inches="tight")
plt.show()

#%% Testing with actual dataset
df_test = pd.read_csv(os.path.join(os.getcwd(), "datasets", "cases_malaysia_test.csv"))
print(df_test.head())

df_test["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
df_test["date"] = df_test["date"].dt.strftime("%Y-%m-%d")

#%%
print(df_test.head())
#%%
df_test["cases_new"] = pd.to_numeric(df_test["cases_new"], errors="coerce")
df_test["cases_new"] = df_test["cases_new"].interpolate(method="polynomial", order=2)

dataset_tot = pd.concat((df, df_test))
dataset_tot = dataset_tot["cases_new"].values

#%%
# MinMaxScaler for X
dataset_tot = mms.transform(np.expand_dims(dataset_tot, axis=-1))

X_actual, y_actual = [], []

for i in range(len(df), len(dataset_tot)):
    X_actual.append(dataset_tot[i - win_size : i])
    y_actual.append(dataset_tot[i])

X_actual = np.array(X_actual)
y_actual = np.array(y_actual)

y_pred_actual = model.predict(X_actual)

#%%
plt.figure()
plt.plot(y_pred_actual, color="red")
plt.plot(y_actual, color="blue")
plt.legend(["Predicted Covid Cases", "Actual Covid Cases"])
plt.savefig("pictures/Imran_predicted_vs_actual.png", dpi=300, bbox_inches="tight")
plt.show()

#%%
print(f"Mean Absolute Error(MSE): {mean_absolute_error(y_actual, y_pred_actual)}")
print(
    f"Mean Absolute Percentage Error(MAPE): {mean_absolute_percentage_error(y_actual, y_pred_actual)}"
)
print(f"r2 Score: {r2_score(y_actual, y_pred_actual)}")

#%% Step 8) Model Deployment

# create directory
dir_name = "model_and_pickle"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# save model
model.save(os.path.join(dir_name, "model.h5"))

# save scaler object
with open(os.path.join(dir_name, "mms.pkl"), "wb") as f:
    pickle.dump(mms, f)
