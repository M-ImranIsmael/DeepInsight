#%% Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import pickle
import datetime
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import plot_model

#%% Step 1) Data Loading

CSV_PATH = os.path.join(os.getcwd(), "dataset", "Dataset.csv")
df = pd.read_csv(CSV_PATH)

#%% Step 2) EDA

# Numerical EDA
print(df.head(3))
print(df.info())
print(df.describe().T)
print(df.shape)

#%% Step 3) Data Cleaning

# Dropping column 'Unnamed: 0' which is just the index

df.drop(df.columns[0], axis=1, inplace=True)
df.head(1)

#%% Categorical VS continuous column

df.nunique()
columns = df.columns
categorical_columns = []
continuous_columns = []

for col in columns:
    if col not in "price":
        if df[col].nunique() <= 22:
            categorical_columns.append(col)
        else:
            continuous_columns.append(col)


print(f"Categorical Columns: {categorical_columns}")
print(f"Continuous Columns: {continuous_columns}")

#%% Handling missing values denoted with '?'

for col in categorical_columns:
    df[col].replace(to_replace="?", value=df[col].mode()[0], inplace=True)

for col in continuous_columns:
    df[col].replace("?", np.nan, inplace=True)
    # Check if the column has any non-integer values
    if (df[col].astype(float) % 1 != 0).any():
        df[col] = pd.to_numeric(df[col], downcast="float")
    else:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    # Impute NaNs with Median
    df[col].fillna(df[col].median(), inplace=True)


print(df.info())

#%% Step 4 Feature Selection (Reselect Features after baseline)

# Cramer's V
def cramers_corrected_stat(confusion_matrix):
    """calculate Cramers V statistic for categorial-categorial association.

    uses correction from Bergsma and Wicher,

    Journal of the Korean Statistical Society 42 (2013): 323-328

    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


# Correlation between categorical and continuous variable
for i in categorical_columns:
    print(i)
    matrix = pd.crosstab(df[i], df["price"]).to_numpy()
    print(cramers_corrected_stat(matrix))

# Correlation between continuous and continuous target variable

plt.figure(figsize=(20, 16))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

#%% Selected Features
to_drop = ["compression-ratio"]
continuous_columns = [col for col in continuous_columns if col not in to_drop]
df = df.drop(to_drop, axis=1)
print(df.columns)
#%% Step 5) Data Preprocessing

# Create the folder if it does not exist
if not os.path.exists("pickle_files"):
    os.makedirs("pickle_files")

label_encoder = LabelEncoder()  # For categorical columns
for col in categorical_columns:
    if df[col].dtype == "object":
        df[col] = label_encoder.fit_transform(df[col])
        print(label_encoder.inverse_transform(df[col].unique()), df[col].unique())
        with open(os.path.join("pickle_files", col + "_le.pkl"), "wb") as f:
            pickle.dump(label_encoder, f)

#%%
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
ss = StandardScaler()
X = ss.fit_transform(X)
# Only apply to regression problem
y = np.expand_dims(y, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.3,
    random_state=42,
)

#%% Step 6) Model Development

input_shape = np.shape(X_train)[1:]
class_num = np.shape(y_train)[1:]

print(input_shape)
print(class_num)

model = Sequential()
model.add(Input(shape=input_shape))
model.add(Dense(1024, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(class_num[0], activation="relu"))
model.summary()

plot_model(model)

model.compile(loss="mse", optimizer="adam", metrics=["mse"])

#%% Tensorboard and training

parent_folder = "tensorboard_log"
log_dir = os.path.join(
    os.getcwd(), parent_folder, datetime.datetime.now().strftime("%Y&m%d-%H%M%S")
)
tb_callback = TensorBoard(log_dir=log_dir)

# To include early stopping to prevent overfitting
# es_callback = EarlyStopping(monitor="loss", patience=3)

hist = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=1500,
    callbacks=[tb_callback],
)

model.evaluate(X_test, y_test)
# %% #* Step 7 Model Analysis

# For regression problems need to use mean_absolute_error, absolute_percentage_error

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
