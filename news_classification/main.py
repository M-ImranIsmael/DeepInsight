"""We're going to try out our model now!"""
#%%
import re
import json
import pickle

from tensorflow import keras
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
from keras.models import load_model

from module import clean_text

# Import the one-hot-encoder (ohe) model and tokenizer
with open("ohe.pkl", "rb") as f:
    loaded_ohe = pickle.load(f)

with open("tokenizer.json", "r") as json_f:
    loaded_tokenizer = json.load(json_f)

# Load the saved neural network model
loaded_model = load_model("model.h5")

# Step 1: Load User Input
# Get input from the user in the form of a list with one element
user_input = [input("Input your news: ")]

# Step 2: Clean Data
# Remove HTML tags, numbers, and standardize cases for the user input

clean_text(user_input)

# Step 3: Preprocess Data
# Transform the user input into a format suitable for the model to make a prediction
# 1. Tokenize the user input
# 2. Pad the sequences to a uniform length
# 3. One-hot-encode the sequences
loaded_tokenizer = tokenizer_from_json(loaded_tokenizer)
user_input = loaded_tokenizer.texts_to_sequences(user_input)
user_input = pad_sequences(user_input, maxlen=350, truncating="post", padding="post")

# Step 4: Make Prediction
# Show a summary of the model's architecture
loaded_model.summary()
# Use the loaded model to make a prediction on the user input
results = loaded_model.predict(user_input)
# Print the predicted results by using the inverse_transform method on the loaded one-hot-encoder
print(loaded_ohe.inverse_transform(results))
