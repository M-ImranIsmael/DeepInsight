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
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard

#%% Constants
# Datasets Path

CSV_PATH = os.path.join(os.getcwd(), "dataset", "gemini_BTCUSD_2020_1min_train.csv")
print(CSV_PATH)

#%% Step 1) Data Loading

df = pd.read_csv(CSV_PATH)

#%% Step 2) EDA

# Numerical EDA

print(df.head())
print(df.info())
print(df.describe().T)
print(df.shape)

#%% Step 3) Data Cleaning

# Order by ascending date
df = df[::-1]
df.head()

#%%
df["Open"] = pd.to_numeric(
    df["Open"], errors="coerce"
)  # Anything thats not a number will be NaN
df.info()

plt.figure(figsize=(30, 10))
plt.plot(df["Open"].values)
plt.show()

# Interpolation to Handle missing values
df["Open"] = df["Open"].interpolate(method="polynomial", order=2)

plt.figure(figsize=(30, 10))
plt.plot(df["Open"].values)
plt.show()

#%% Step 4) Feature Selections

data = df["Open"].values  # To obtain data in ndarray format

#%% Step 5) Data Preprocessing
mms = MinMaxScaler()
data = mms.fit_transform(np.expand_dims(data, axis=-1))

#%%
X_train, y_train = [], []  # Instantiate empty list

win_size = 60

for i in range(win_size, len(data)):
    X_train.append(data[i - win_size : i])
    y_train.append(data[i])

# Convert list into ndarray
X_train = np.array(X_train)
y_train = np.array(y_train)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.3, random_state=123, shuffle=True
)
#%% Step 6) Model Development

input_shape = np.shape(X_train)[1:]
model = Sequential()
model.add(SimpleRNN(64, activation="tanh", input_shape=input_shape))
model.add(Dropout(0.3))
model.add(Dense(1, activation="linear"))
model.summary()
plot_model(model, show_shapes=True)

model.compile(optimizer="adam", loss="mse", metrics="mse")

#%% Tensorboard and fitting

parent_folder = "tensorboard_log"
log_dir = os.path.join(
    os.getcwd(), parent_folder, datetime.datetime.now().strftime("%Y&m%d-%H%M%S")
)
tb_callback = TensorBoard(log_dir=log_dir)

# To include early stopping to prevent overfitting

es_callback = EarlyStopping(monitor="loss", patience=3)

hist = model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[tb_callback, es_callback],
)

#%% # Step 7 Model Analysis
plt.figure()
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.legend(["training loss", "validation loss"])
plt.show()

plt.figure()
plt.plot(hist.history["mse"])
plt.plot(hist.history["val_mse"])
plt.legend(["training accuracy", "validation accuracy"])
plt.show()

#%%
y_true = y_test
y_pred = model.predict(X_test)
print(mean_absolute_error(y_true, y_pred))
print(mean_absolute_percentage_error(y_true, y_pred))
print(r2_score(y_true, y_pred))
#%% Step 8) Model Deployment

#%% Step 8 Model Deployment

model.save("model.h5")
