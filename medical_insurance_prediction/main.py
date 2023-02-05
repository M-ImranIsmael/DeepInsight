# We're going to test our model on unseen data now!

#%%
import pickle 
import numpy as np

from tensorflow import keras
from keras.models import load_model


# %%

# Example unseen data
data = [19, 0, 27.9, 0, 1, 3] # Features data for the first row in our insurance datasets

# Loading our pickle files to decode them
with open('pickle_files/ss.pkl', 'rb') as f:
    loaded_ss = pickle.load(f)

with open('pickle_files/sex_le.pkl', 'rb') as f:
    sex_le = pickle.load(f)

with open('pickle_files/smoker_le.pkl', 'rb') as f:
    smoker_le = pickle.load(f)

with open('pickle_files/region_le.pkl', 'rb') as f:
    region_le = pickle.load(f)

data = loaded_ss.transform(np.expand_dims(data, axis=0))
loaded_model = load_model('model.h5')

loaded_model.summary()


# %%  Model prediction

print(f'Insurance prediction for the data ($): {loaded_model.predict(data)}')

'''
The model's prediction for the insurance cost for this data point is around 20,000 dollars, which is 4,000 dollars higher than the original data. This suggests that the model is making accurate predictions and is performing well
'''

