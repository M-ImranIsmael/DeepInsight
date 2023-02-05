#%% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle 
import os

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
CSV_PATH = os.path.join(os.getcwd(), 'dataset', 'Dry_Bean_Dataset.csv')
df = pd.read_csv(CSV_PATH)

#%% Step 2) EDA

# Numerical EDA
print(df.head(3))
print(df.info())
print(df.describe().T)
print(df.shape)

#%% Step 3) Data Cleaning

# Checking for missing values
print(df.isna().sum())

# Checking for duplicates
print(df.duplicated().sum())

# Dropping duplicates
df.drop_duplicates(inplace=True)

#%% Removing outliers

def plot_con(df):
    df.plot(kind="box", subplots=True, sharey=False, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.5)
    plt.show()

plot_con(df)

for x in df.columns:
    if x != "Class":
        df[x] = df[x].clip(lower=df[x].quantile(0.01), upper=df[x].quantile(0.99))

plot_con(df)

#%% Step 4) Feature Selections

# Correlation analysis continuous vs Class
continuous_columns = df.drop(labels='Class', axis=1)
print(continuous_columns.columns)

for col in continuous_columns.columns:
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(np.expand_dims(df[col], axis=1), df['Class'])
    print(col)
    print(lr.score(np.expand_dims(df[col], axis=1), df['Class']))

#%% Selected Features (Run cell after baseline results w/o feature selection)
selected_features = continuous_columns.drop(labels=['ConvexArea', 'Area'], axis=1)

#%% Step 5) Data Preprocessing

X = selected_features
y = df['Class']

ss = StandardScaler()  
X = ss.fit_transform(X)  # Standardize features and converting X to ndarray

ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(np.expand_dims(y, axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%% Step 6) Model Development

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

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=10, callbacks=[tb_callback, es_callback])
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

labels = ohe.inverse_transform(np.unique(y, axis=0))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=90)
plt.show()

#%% Step 8) Model Deployment

model.save('model.h5')

with open('ohe.pkl', 'wb') as file:
    pickle.dump(ohe, file)
