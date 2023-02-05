# RNN AND LSTM

RNN (Recurrent Neural Network) and LSTM (Long-Short-Term Memory) are two types of deep learning algorithms used for sequence data analysis.

RNNs are designed to process sequential information, where the output of one step is used as input for the next step. However, they are limited in their ability to preserve information over longer sequences due to the vanishing gradient problem.

LSTMs, on the other hand, are a type of RNN that are specifically designed to overcome the vanishing gradient problem by introducing a memory cell, gates to control the flow of information, and the ability to maintain information for a longer period of time.

LSTMs have been successful in various sequence data analysis tasks such as language modeling, machine translation, and speech recognition.

## EDA in Time-Series

- When doing df.describe().T in time-series data, it is not really suitable as it's a sequential data rather than the one previously where each data is independent

- To remove outliers, we used different method

## Best graph to visualize time-series data:

- Line plot:

  - If there's a string inside a supposed float dtype, then plotting will looked weird
  - Try to do data cleaning first:

          df['Open'] = pd.to_numeric(df['Open'], errors='coerce')  # Anything thats not a number will be NaN

- Scatter Plot

## Handling missing values

1.  Dropping the missing values (dropna):

    a) accuracy will drop since data is now dependent

    b) the graph will just join

    c) bad method

2.  Simple imputation (impute with median/mean):

    a) Bad method as all NaN will fluctuates to the median/mean

3.  Backward fill and forward fill:

    a) Good in some cases eg: (medical cases: good with forward fill)

    b) Forward fill:

         df['Open'].fillna(methods='ffill')

    c) But what if we have a lot of missing values? This won't look nice later in the graph

4.  Interpolation:

         df['Open'] = df['Open'].interpolate(method='polynomial', order=2)

    a) If there's a lot of NaNs, its always better to use interpolation rather than backward/forward fill

## Data Processing

- for stock price it is suitable to use MinMaxScaler
- No X and Y
- We're taking

## Model Development

- No batch normalization, and remember X needs to be 3D

- If we want to add more hidden layers we cannot used the method before:
  needs return_sequence=True
