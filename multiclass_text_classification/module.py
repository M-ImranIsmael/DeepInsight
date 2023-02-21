import re
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras import Sequential

#%%
def clean_text(text):
    for index, data in enumerate(text):
        # Replace urls with a space
        temp = re.sub("www\.[\w.]+", " ", data)
        temp = re.sub("[\w.]+\.com", " ", temp)

        # Replacing all characters except letters with a space
        temp = re.sub("[^a-zA-Z]", " ", temp)

        # Change don t to dont
        temp = re.sub("[^ \na-zA-Z]", "", temp)

        text[index] = temp.lower()
    return text


def model_lstm(vocab_size, class_num, embedding_dim=64, dropout_rate=0.3):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))  # 5000, 64
    model.add(LSTM(embedding_dim, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(embedding_dim))
    model.add(Dropout(dropout_rate))
    model.add(Dense(class_num, activation="softmax"))  # Output layer
    model.summary()
    return model


def plot_hist(hist):
    keys = hist.history.keys()
    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(["loss", "accuracy"]):
        ax = plt.subplot(1, 2, i + 1)
        ax.plot(hist.history[metric], color="maroon", linestyle="-")
        ax.plot(hist.history[f"val_{metric}"], color="purple", linestyle="--")
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel(metric.capitalize(), fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(True)
        ax.legend(["Training", "Validation"], fontsize=12)
    plt.tight_layout()
    plt.show()
