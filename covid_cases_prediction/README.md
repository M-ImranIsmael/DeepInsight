# Covid-19 Daily Cases Prediction in Malaysia

This project aims to forecast the number of Covid-19 daily cases in Malaysia. The daily cases data are obtained from the [Ministry of Health Malaysia's GitHub Repo](https://github.com/MoH-Malaysia/covid19-public).

## Build With:

<p align="left">  <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a>

## Directory Structure

1. [datasets](https://github.com/M-ImranIsmael/Deep_Learning_Applications/tree/master/covid_cases_prediction/datasets):

   - cases_malaysia_test.csv
   - cases_malaysia_train.csv

2. [training.py](https://github.com/M-ImranIsmael/Deep_Learning_Applications/blob/master/covid_cases_prediction/imran_training.py):

   - Importing necessary libraries
   - Loading and cleaning the data and EDA
   - Model selection and training using LSTM model
   - Model saving and testing on new datasets

3. [model_and_pickle](https://github.com/M-ImranIsmael/Deep_Learning_Applications/tree/master/covid_cases_prediction/model_and_pickle): trained deep learning model and pickle file for mms

4. [pictures](https://github.com/M-ImranIsmael/Deep_Learning_Applications/tree/master/covid_cases_prediction/pictures): plots and results

## Results

### Daily New COVID-19 Cases in Malaysia Plot

![alt text](pictures/Imran_new_cases_plot.png)

### LSTM Model Architecture

![alt text](pictures/Imran_model_architecture.png)

### Tensorboard Result

![alt text](pictures/Imran_tensorboard_epochloss.png)
![alt text](pictures/Imran_tensorboard_epochmse.png)

### Actual vs Predicted Covid-19 Cases

![alt text](pictures/Imran_predicted_vs_actual.png)

### Model Reports

![alt text](pictures/Imran_mse_mape_r2score.png)

## Acknowledgment of Data

The dataset used in this project was obtained from:
[Ministry of Health Malaysia's GitHub Repo](https://github.com/MoH-Malaysia/covid19-public).
