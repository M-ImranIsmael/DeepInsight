#%% Libraries
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
)

from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import plot_model

#%% Step 1) Data Loading

CSV_PATH = os.path.join(os.getcwd(), 'dataset', 'cardio_train.csv')
df = pd.read_csv(CSV_PATH, sep=';')

#%% Step 2) EDA

# Numerical EDA
print(df.head(3))
print(df.info())
print(df.describe().T)
print(df.shape)

#%% Step 3) Data Cleaning

# Dropping ID column
df = df.drop(labels='id', axis=1)

# Changing age column from days to year
df['age'] = round(df['age']/365.25)
df['age'] = df['age'].astype(int)

# Checking for missing values
print(df.isna().sum())

# Dropping missing values
df.dropna(inplace=True)

#%% Categorical VS continuous column

df.nunique()
columns = df.columns
categorical_columns = []
continuous_columns = []

for col in columns:
    if col not in 'cardio':
        if df[col].nunique() <= 4:  # Since our highest unique value is 3
            categorical_columns.append(col)
        else:
            continuous_columns.append(col)

print(f'Categorical columns: {categorical_columns}')
print(f'Continuous columns: {continuous_columns}')


#%% Visual EDA
palettes = ["deep", "muted", "pastel", "bright", "dark"]

for i, x in enumerate(categorical_columns):
    # sns.color_palette("husl", 8)
    sns.countplot(data=df, x=x)
    plt.xlabel(x)
    plt.show()



#%% Removing outliers

df.boxplot(continuous_columns)
plt.show()

# Replacing outliers with NaNs by using IQR
for x in continuous_columns:
    q75, q25 = np.percentile(df.loc[:, x], [75, 25])
    intr_qr = q75 - q25

    maximum = q75 + (1.5 * intr_qr)
    minimum = q25 - (1.5 * intr_qr)

    df.loc[df[x] < minimum, x] = np.nan
    df.loc[df[x] > maximum, x] = np.nan

df = df.dropna(axis=0)

df.boxplot(continuous_columns)
plt.show()

#%% Step 4) Feature Selections

#Correlation analysis continuous vs cardio
for i in continuous_columns:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[i], axis=1), df["cardio"])
    print(i)
    print(lr.score(np.expand_dims(df[i], axis=1), df["cardio"]))

#Correlation analysis categorical vs cardio

#Cramer's V
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

for i in categorical_columns:
    cm = pd.crosstab(df[i], df["cardio"]).to_numpy()
    print(i)
    print(cramers_corrected_stat(cm))


#%% Selected Features (Run cell after baseline results w/o feature selection)
selected_features = continuous_columns + ['cardio']
print(selected_features)

#%% Step 5) Data Preprocessing
X = df.drop(labels='cardio', axis=1)
y = df['cardio']

ss = StandardScaler()  
X = ss.fit_transform(X)  # Standardize features and converting X to ndarray

ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(np.expand_dims(y, axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%% Step 6) Model Development using Functional API in TF

input_shape = np.shape(X_train)[1:]
num_class = len(np.unique(y, axis=0))
hidden_layers = [32, 16]

print(input_shape)
print(num_class)

input_layer = Input(shape=input_shape)
hidden = input_layer

for i, neurons in enumerate(hidden_layers):
    hidden = Dense(neurons, activation="relu", name="hidden_layer_" + str(i + 1))(
        hidden
    )
    hidden = BatchNormalization()(hidden)
    hidden = Dropout(0.2)(hidden)
output = Dense(num_class, activation="softmax")(hidden)

model = Model(inputs=input_layer, outputs=output)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

plot_model(model, show_shapes=True)


#%% Tensorboard and Training

parent_folder = 'tensorboard_log'
log_dir = os.path.join(os.getcwd(), parent_folder, datetime.datetime.now().strftime("%Y&m%d-%H%M%S"))

tb_callback = TensorBoard(log_dir=log_dir)
es_callback = EarlyStopping(monitor='loss', patience=3) 

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, callbacks=[tb_callback, es_callback])
model.evaluate(X_test, y_test)

#%% Step 7 Model Analysis
plt.figure()
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.legend(["training loss", "validation loss"])
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.legend(["training loss", "validation loss"])
plt.show()

# %% Evaluate model using confusion matrix
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)  # get the highest value
y_test = np.argmax(y_test, axis=1)


# %% To display confusion matrix
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print(cr)

labels = ["cvd absence", "cvd presence"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#%% Step 8) Model Deployment