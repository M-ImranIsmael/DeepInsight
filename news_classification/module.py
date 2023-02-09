import re
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.layers import Dense, Dropout, LSTM, Embedding, GRU
from keras import Sequential


def clean_text(text):
    for index, data in enumerate(text):
        # Replacing 'bit.ly/<numbers>/<words>' with a space
        temp = re.sub("bit.ly/\d+/\w{1,10}", " ", data)
        # Replacing '[<number> <word>]' with a space
        temp = re.sub("\[\d+\s*\w+]", " ", temp)
        # Replacing 'www.<words>.<words>' with a space
        temp = re.sub("www\.[\w.]+", " ", temp)
        # Replacing all characters except letters with a space
        temp = re.sub("[^a-zA-Z]", " ", temp)
        # Change don t to dont
        temp = re.sub("[^ \na-zA-Z]", "", temp)
        text[index] = temp.lower()
    return text


def simple_lstm(vocab_size, class_num, embedding_dim=64, dropout_rate=0.3):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))  # 5000, 64
    model.add(LSTM(embedding_dim, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(embedding_dim))
    model.add(Dropout(dropout_rate))
    model.add(Dense(class_num, activation="softmax"))  # Output layer
    model.summary()
    return model


def simple_gru(vocab_size, class_num, embedding_dim=64, dropout_rate=0.3):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))  # 5000, 64
    model.add(GRU(embedding_dim))
    model.add(Dropout(dropout_rate))
    model.add(Dense(class_num, activation="softmax"))  # Output layer
    model.summary()
    return model


def plot_hist(hist):
    keys = hist.history.keys()
    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(["loss", "accuracy"]):
        plt.subplot(1, 2, i + 1)
        plt.plot(hist.history[metric])
        plt.plot(hist.history[f"val_{metric}"])
        plt.title(metric)
        plt.legend(["Training", "Validation"])
    plt.show()
