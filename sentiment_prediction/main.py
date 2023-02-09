"""We're going to try out our model now!"""

#%% Libraries
import re
import json
import pickle

import re
import json
import pickle

from tensorflow import keras
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
from keras.models import load_model

# Import the one-hot-encoder (ohe) model and tokenizer
with open("ohe.pkl", "rb") as f:
    loaded_ohe = pickle.load(f)

with open("tokernizer.json", "r") as json_f:
    loaded_tokenizer = json.load(json_f)

# Load the saved neural network model
loaded_model = load_model("model.h5")

# Step 1: Load User Input
# Get input from the user in the form of a list with one element
user_input = [input("Write your review for this movie: ")]

# Step 2: Clean User's Input Data
# Remove HTML tags, numbers, and standardize cases for the user input
for index, data in enumerate(user_input):
    # Removing HTML tags and numbers, standardizing cases
    data = re.sub("<.*?>", "", data)
    data = re.sub("[^a-zA-Z]", " ", data).lower()
    user_input[index] = data

# Step 3: Preprocess Data
# Transform the user input into a format suitable for the model to make a prediction
# 1. Tokenize the user input
# 2. Pad the sequences to a uniform length
# 3. One-hot-encode the sequences

loaded_tokenizer = tokenizer_from_json(loaded_tokenizer)
user_input = loaded_tokenizer.texts_to_sequences(user_input)
pad_sequences(user_input, maxlen=350, truncating="post", padding="post")

# Step 4: Make Prediction
# Show a summary of the model's architecture
loaded_model.summary()

# Use the loaded model to make a prediction on the user input
results = loaded_model.predict(user_input)
print(loaded_ohe.inverse_transform(results))
