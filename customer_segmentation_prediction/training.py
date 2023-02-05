#%%
#%% Libraries
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import pickle

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
)

from tensorflow import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import plot_model

#%%
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

#%% # * Step 1 Data Loading
df = pd.read_csv('datasets/train.csv')


#%% # * Step 2 EDA
print(df.head())
print(df.info())
print(df.shape)
print(df.describe(include='all').T)

"""
Through df.info and df.describe we can see that there are NaN values
"""

#%% # *  Step 3 Data Cleaning

# Dropping ID columns

df = df.drop(labels='ID', axis=1)


#%%
df.nunique()
columns = df.columns
categorical_columns = []
continuous_columns = []
for col in columns:
    if col != "Segmentation":
        if col != 'Family_Size' and df[col].nunique() <= 9:
            categorical_columns.append(col)
        else:
            continuous_columns.append(col)

print("Categorical Columns: ", categorical_columns)
print("Continuous Columns: ", continuous_columns)
#%% # * Handling missing values

print(df.isna().sum())

# * Method 1 Dropping missing values
df = df.dropna()
print(df.isna().sum())
print(df.shape)

# for column in categorical_columns:
#     df[column].fillna(df[column].mode()[0], inplace=True)
    
# for column in continuous_columns:
#     imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean')
#     df[column] = imputer.fit_transform(df[[column]])
    
# print(df.head())
# print(df.isna().sum())

#%% # * Handling outliers



#%% # * Plotting boxplot to identify outliers
df.boxplot(continuous_columns)
#%%
# * Step 4 Feature Selection
for i in categorical_columns:
    cm = pd.crosstab(df[i], df["Segmentation"]).to_numpy()
    print(i)
    print(cramers_corrected_stat(cm))

#%%
for i in continuous_columns:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[i], axis=1), df["Segmentation"])
    print(i)
    print(lr.score(np.expand_dims(df[i], axis=1), df["Segmentation"]))

categorical_columns = ['Ever_Married', 'Graduated', 'Profession', 'Spending_Score']


#%% # * Step 5 Data Preprocessing
ohe = OneHotEncoder(sparse_output=False)
X_categorical = ohe.fit_transform(df[categorical_columns])
X_continuous = df[continuous_columns]
scaler = StandardScaler()
X_continuous = scaler.fit_transform(X_continuous)
X = np.concatenate((X_continuous, X_categorical), axis=1)
y = df['Segmentation']
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(np.expand_dims(y, axis=1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#%%
#%5 # * Step 6 Model Development using functional
input_shape = np.shape(X_train)[1:]
nb_class = len(np.unique(y_train, axis=0))
hidden_layers = [256, 128, 64, 32, 16]
input_layer = Input(shape=input_shape)

hidden = input_layer
for i, neurons in enumerate(hidden_layers):
    hidden = Dense(neurons, activation="relu", name="hidden_layer_" + str(i + 1))(
        hidden
    )
    hidden = BatchNormalization()(hidden)
    hidden = Dropout(0.2)(hidden)
output = Dense(nb_class, activation="softmax")(hidden)

model = Model(inputs=input_layer, outputs=output)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

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
    validation_data=(X_test, y_test),
    # batch_size=128,
    epochs=20,
    callbacks=[tb_callback, es_callback]
)

model.evaluate(X_test, y_test)

#%%
# * Step 7 Model Analysis

plt.figure()
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.legend(["training loss", "validation loss"])
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.legend(["training accuracy", "validation accuracy"])
plt.show()
# %%
