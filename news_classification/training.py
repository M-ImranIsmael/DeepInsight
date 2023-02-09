#%% Libraries
import pandas as pd
import numpy as np
import datetime
import pickle
import json
import os

from tensorflow import keras
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, TensorBoard

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from module import plot_hist, simple_lstm, simple_gru, clean_text

#%% Constants

CSV_PATH = os.path.join(os.getcwd(), "dataset", "True.csv")  # Path to csv file

#%% Step 1) Data Loading

df = pd.read_csv(CSV_PATH)  # Reading the data into dataframe

#%% Step 2) EDA

# Viewing the contents of the first 3 and last 3 of the df
print(df.head(3))
print(df.tail(3))

#%% Step 3) Data Cleaning

text = df["text"]  # Feature
subject = df["subject"]  # Target
clean_text(text)  # Calling function from module.py


#%% Step 4) Feature Selections

# Nothing to select

#%% Step 5) Data Preprocessing

# Text Vectorization/ Tokenization

# Setting the vocabulary size to 5000
vocab_size = 5000

# Creating a token for words that are not in the vocabulary
oov_token = "<00V>"

# Initializing the tokenizer with the vocabulary size and out of vocabulary token
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

# Training the tokenizer with the cleaned text data
tokenizer.fit_on_texts(text)

# Retrieving the word index of the trained tokenizer
word_index = tokenizer.word_index

# Printing the first 10 word indexes to confirm the training is done correctly
print(list(word_index.items())[0:10])

# Converting the cleaned text data into sequences of integers
text_tokenized = tokenizer.texts_to_sequences(text)

#%%
# Calculating the length of each tokenized text
max_length = [len(i) for i in text_tokenized]

# Determining the maximum length by taking the 75th percentile of the lengths
max_length = int(np.ceil(np.percentile(max_length, 75)))

# Padding and truncating the tokenized text data
text_tokenized = pad_sequences(
    text_tokenized, maxlen=max_length, padding="post", truncating="post"
)

#%%
# One hot encoding for target
ohe = OneHotEncoder(sparse_output=False)
subject = ohe.fit_transform(np.expand_dims(subject, axis=1))


#%% Train test split

text_tokenized = np.expand_dims(
    text_tokenized, axis=-1
)  # Adding a dimension to the tokenized text data to meet the 3D input requirement for LSTM/RNN/GRU models

X_train, X_test, y_train, y_test = train_test_split(
    text_tokenized, subject, test_size=0.3, shuffle=True, random_state=123
)
#%% Step 6) Model Development

class_num = len(np.unique(y_train, axis=0))

# Calling simple_gru function from module.py
model = simple_gru(vocab_size, class_num)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="accuracy")

#%% Tensorboard callbacks and fitting

parent_folder = "tensorboard_log"
log_dir = os.path.join(
    os.getcwd(), parent_folder, datetime.datetime.now().strftime("%Y&m%d-%H%M%S")
)

tb_callback = TensorBoard(log_dir=log_dir)
es_callback = EarlyStopping(monitor="loss", patience=3)

hist = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    callbacks=[tb_callback, es_callback],
)


#%% Model Analysis

# Calling pot_hist function from module.py
plot_hist(hist)

#%%
# Getting predictions from the model on the test set
y_pred = model.predict(X_test)
y_true = y_test

# Extracting the class with the highest probability from the predictions
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_true, axis=1)

# Printing the classification report, confusion matrix and accuracy score
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))

#%% Step 8) Model Deployment

model.save("model.h5")  # Save the trained deep learning model to directory

with open("ohe.pkl", "wb") as f:  # Open the file "ohe.pkl" in write binary mode
    pickle.dump(ohe, f)  # Write the one-hot-encoding object to the file

# Convert the tokenizer object to a JSON string
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w") as json_file:
    json.dump(tokenizer_json, json_file)
# %%
