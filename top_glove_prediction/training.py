#%% #*Libraries
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
from keras.layers import Dense, Dropout, SimpleRNN
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard

CSV_PATH = os.path.join(
    os.getcwd(), "dataset", "Top_Glove_Stock_Price_Train(Modified).csv"
)


#%% Step 1) Data Loading
df = pd.read_csv(CSV_PATH)
column_names = df.columns
print(column_names)


#%% Step 2) EDA

print(df.head(3))
print(df.info())
print(df.describe())
print(df.shape)

plt.figure()
plt.plot(df["Open"])
plt.show()


#%% Step 3) Data Cleaning

# Interpolation to Handle missing values
df["Open"] = df["Open"].interpolate(method="polynomial", order=2)

plt.figure()
plt.plot(df["Open"])
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


#%% #* Step 6) Model Development

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
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[tb_callback, es_callback],
)


#%% #* Step 7) Model Analysis

print(hist.history.keys())
plt.figure()
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.legend(["Training Loss", "Validation Loss"])
plt.show()


#%% Testing with actual test

df_test = pd.read_csv(
    os.path.join(os.getcwd(), "dataset", "Top_Glove_Stock_Price_Test.csv"),
    names=column_names,
)

dataset_tot = pd.concat((df, df_test))
dataset_tot = dataset_tot["Open"]

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
plt.legend(["Predicted Stock Price", "Actual Stock Price"])
plt.show()

#%%
print(mean_absolute_error(y_actual, y_pred_actual))
print(mean_absolute_percentage_error(y_actual, y_pred_actual))
print(r2_score(y_actual, y_pred_actual))


#%% Step 8) Model Deployment

model.save("model.h5")

with open("mms.pkl", "wb") as f:
    pickle.dump(mms, f)
