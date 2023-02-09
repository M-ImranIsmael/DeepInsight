#%% Libraries
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pickle
import json
import re
import os

from tensorflow import keras
from keras import Sequential

from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import pad_sequences

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#%% Step 1) Data Loading

df = pd.read_csv(
    "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
)

#%% Step 2) EDA

# Viewing the contents of the first 3 and last 3 of the df
print(df.head(3))
print(df.tail(3))

#%% Step 3) Data Cleaning

review = df["review"]
sentiment = df["sentiment"]

for index, data in enumerate(review):
    # Removing HTML tags and numbers, standardizing cases
    data = re.sub("<.*?>", "", data)
    data = re.sub("[^a-zA-Z]", " ", data).lower()
    review[index] = data

#%% Step 4) Feature Selections

# Nothing to select!

#%% Step 5) Data Preprocessing

# Text Vectorization/ Tokenization

vocab_size = 5000  # Setting the vocabulary size to 5000
oov_token = "<00V>"  # Initializing the tokenizer with the vocabulary size and out of vocabulary token

# Initializing the tokenizer with the vocabulary size and out of vocabulary token
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

# Training the tokenizer with the cleaned reviews
tokenizer.fit_on_texts(review)

# Retrieving the word index of the trained tokenizer
word_index = tokenizer.word_index

# Printing the first 10 word indexes to confirm the training is done correctly
print(list(word_index.items())[0:10])

# Converting the cleaned reviews data into sequences of integers
review_tokenized = tokenizer.texts_to_sequences(review)

#%%
# Calculating the length of each tokenized reviews
max_length = [len(i) for i in review_tokenized]

max_length = int(np.ceil(np.median(max_length)))

# Padding and truncating the tokenized reviews
review_tokenized = pad_sequences(
    review_tokenized, maxlen=350, padding="post", truncating="post"
)

#%%
# One hot encoding for target
ohe = OneHotEncoder(sparse_output=False)
sentiment = ohe.fit_transform(np.expand_dims(sentiment, axis=1))

#%% Train test split

# Adding a dimension to the tokenized text data to meet the 3D input requirement for LSTM/RNN/GRU models
review_tokenized = np.expand_dims(review_tokenized, axis=-1)
X_train, X_test, y_train, y_test = train_test_split(
    review_tokenized, sentiment, test_size=0.3, shuffle=True, random_state=123
)

#%% Step 6) Model Development

# input_shape = np.shape(X_train)[1:]
embedding_dim = 64

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))  # 5000, 64
model.add(LSTM(embedding_dim, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(2, activation="softmax"))  # Output layer
model.summary()

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

#%% Step 7) Model Analysis

keys = hist.history.keys()
plt.figure(figsize=(10, 5))
for i, metric in enumerate(["loss", "accuracy"]):
    plt.subplot(1, 2, i + 1)
    plt.plot(hist.history[metric])
    plt.plot(hist.history[f"val_{metric}"])
    plt.title(metric)
    plt.legend(["Training", "Validation"])
plt.show()

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

# Save the trained deep learning model to directory
model.save("model.h5")

# Saving one-hot-encoding target labels
with open("ohe.pkl", "wb") as f:
    pickle.dump(ohe, f)

# Saving tokenizer
tokenizer_json = tokenizer.to_json()
with open("tokernizer.json", "w") as json_file:
    json.dump(tokenizer_json, json_file)
