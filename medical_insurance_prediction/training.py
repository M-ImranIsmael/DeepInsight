#%% Libraries
import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import datetime

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping

#%% Step 1) Data Loading

CSV_PATH = os.path.join(os.getcwd(), "dataset", "insurance.csv")
df = pd.read_csv(CSV_PATH)

#%% Step 2) EDA
# Numerical EDA
print(df.head(3))
print(df.info())
print(df.describe().T)
print(df.shape)

#%% Step 3) Data Cleaning

# Checking for missing values
print(df.isna().sum())  # No missing values

# Checking for duplicates
print(df.duplicated().sum())  # Contains one duplicated value

# Checking for outliers
df.boxplot()  # No outliers

# Dropping duplicates
df.drop_duplicates(inplace=True)

#%% Categorical vs Continuous column

print(df.nunique())
columns = df.columns
categorical_columns = []
continuous_columns = []

for col in columns:
    if col not in "charges":
        if df[col].nunique() <= 6:  # Since our highest unique value is 6
            categorical_columns.append(col)
        else:
            continuous_columns.append(col)

print(f"Categorical columns: {categorical_columns}")
print(f"Continuous columns: {continuous_columns}")


# %% Visual EDA
for cat in categorical_columns:
    plt.figure()
    sns.displot(df[cat])
    plt.show()

for con in continuous_columns:
    plt.figure()
    sns.distplot(df[con])
    plt.show()

#%% Step 4) Feature Selections

# Correlation between continuous and continuous target variable
corr = df.corr()["charges"]
sns.heatmap(df.corr(), annot=True)
plt.show()

#%%
for col in categorical_columns:
    print(f"{col}: {df[col].dtype}")

#%% Step 5) Data Preprocessing

label_encoder = LabelEncoder()
for col in categorical_columns:
    if df[col].dtype == "object":
        df[col] = label_encoder.fit_transform(df[col])
        print(label_encoder.inverse_transform(df[col].unique()))
        with open(os.path.join("pickle_files", col + "_le.pkl"), "wb") as f:
            pickle.dump(label_encoder, f)

#%%
X = df.drop(labels="charges", axis=1)
y = df["charges"]

ss = StandardScaler()
X = ss.fit_transform(X)  # Standardize features and converting X to ndarray
y = np.expand_dims(y, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


#%% Step 6) Model Development using Sequential API in TF
input_shape = np.shape(X_train)[1:]
class_num = np.shape(y_train)[1:]

print(input_shape)
print(class_num)

model = Sequential()
model.add(Input(shape=input_shape))
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(class_num[0], activation="relu"))
model.summary()

plot_model(model)
model.compile(loss="mse", optimizer="adam", metrics=["mse"])

#%% Tensorboard and Training
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
    validation_data=(X_test, y_test),
    epochs=500,
    callbacks=[tb_callback, es_callback],
)

model.evaluate(X_test, y_test)

#%% Step 7) Model Analysis

# For regression problems we can use mean_absolute_error, absolute_percentage_error

y_true = y_test
y_pred = model.predict(X_test)
print(mean_absolute_error(y_true, y_pred))
print(mean_absolute_percentage_error(y_true, y_pred))
print(r2_score(y_true, y_pred))


#%%
plt.figure()
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.legend(["training loss", "validation loss"])
plt.show()

plt.figure()
plt.plot(hist.history["mse"])
plt.plot(hist.history["val_mse"])
plt.legend(["training mse", "validation mse"])
plt.show()

#%% Step 8) Model Deployment

model.save("model.h5")

with open(os.path.join("pickle_files", "ss.pkl"), "wb") as f:
    pickle.dump(ss, f)
