# Libraries
# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

from tensorflow import keras
from keras import Sequential
from keras.layers import SimpleRNN, Dropout, Dense
from keras.utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#%% # Step 1 Data Loading
CSV_PATH = os.path.join(
    os.getcwd(), "dataset", "Google_Stock_Price_Train(modified).csv"
)
df = pd.read_csv(CSV_PATH)

#%% # Step 2 EDA

print(df.head(3))
print(df.info())
print(df.describe().T)
print(df.shape)

#%% # Step 3 Data Cleaning

df["Open"] = pd.to_numeric(
    df["Open"], errors="coerce"
)  # Anything thats not a number will be NaN
df.info()

plt.figure()
plt.plot(df["Open"])
plt.show()

df.isna().sum()

#%% # Handling missing values

# Method 1 Forward fill
df_fill_na = df["Open"].fillna(method="ffill")

plt.figure()
plt.plot(df_fill_na)
plt.show()

# Method 2 Interpolation
df["Open"] = df["Open"].interpolate(method="polynomial", order=2)  # Order 2/3


#%% # Step 4 Feature Selection

data = df["Open"].values

#%% # Step 5 Data Preprocessing

mms = MinMaxScaler()
# data = mms.fit_transform(np.reshape(data, (-1,-1)))
# OR
# data = mms.fit_transform(data.reshape(1, -1))
# OR
data = mms.fit_transform(np.expand_dims(data, -1))

#%%
X_train, y_train = [], []

win_size = 60

for i in range(len(df), len(data)):
    X_train.append(data[i - win_size : i])
    y_train.append(data[i])

# Convert list into array
X_train = np.array(X_train)
y_train = np.array(y_train)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.3, random_state=123
)

#%%
# Step 6 Model Development

input_shape = np.shape(X_train)[1:]
model = Sequential()
model.add(
    SimpleRNN(64, activation="tanh", input_shape=input_shape, return_sequences=True)
)
model.add(Dropout(0.2))
model.add(
    SimpleRNN(64, activation="tanh", input_shape=input_shape, return_sequences=True)
)
model.add(Dropout(0.2))
model.add(SimpleRNN(64, activation="tanh", input_shape=input_shape))
model.add(Dropout(0.2))
model.add(Dense(1, activation="linear"))


model.compile(optimizer="adam", loss="mse", metrics="mse")
model.summary()

plot_model(model, show_shapes=True)

#%% # * Tensorboard callbacks

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


#%% Model Analysis against actual data

df_test = pd.read_csv(os.path.join(os.getcwd(), 'dataset', 'Google_Stock_Price_Test.csv'))

# Joining both datasets (train and test)
dataset_tot = pd.concat((df, df_test))
dataset_tot = dataset_tot['Open']   


# MinMaxScaler for X
dataset_tot = mms.transform(np.expand_dims(dataset_tot, axis=-1))


X_actual, y_actual = [], []


for i in range(len(df), len(dataset_tot)):  
    X_actual.append(dataset_tot[i-win_size: i ])
    y_actual.append(dataset_tot[i])

X_actual = np.array(X_actual)
y_actual = np.array(y_actual)

#%% Testing the model with actual data

y_pred_actual = model.predict(X_actual)

plt.figure()
plt.plot(y_pred_actual, color='red')
plt.plot(y_actual, color='blue')
plt.legend(['Predicted Stock Price', 'Actual Stock Price'])
plt.show()

#%%
y_actual = y_test
print(mean_absolute_error(y_actual, y_pred_actual))
print(mean_absolute_percentage_error(y_actual, y_pred_actual))
print(r2_score(y_actual, y_pred_actual))

#%% Step 8 Model Deployment

model.save("model.h5")
