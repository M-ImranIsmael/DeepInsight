# Covid-19 Daily Cases Prediction in Malaysia

This project aims to forecast the number of Covid-19 daily cases in Malaysia. The daily cases data are obtained from the [Ministry of Health Malaysia's GitHub Repo](https://github.com/MoH-Malaysia/covid19-public).

## Directory Structure:

1. [datasets](https://github.com/M-ImranIsmael/Deep_Learning_Applications/tree/master/covid_cases_prediction/datasets):

   - cases_malaysia_test.csv
   - cases_malaysia_train.csv

2. [training.py](https://github.com/M-ImranIsmael/Deep_Learning_Applications/blob/master/covid_cases_prediction/imran_training.py) which consist of the following steps:

   - Importing necessary libraries
   - Loading and cleaning the data and EDA
     ### Daily New COVID-19 Cases in Malaysia Plot
     ![alt text](pictures/Imran_new_cases_plot.png)
   - Model selection and training

     ### LSTM Model Architecture

     ![alt text](pictures/Imran_model_architecture.png)

     ### Tensorboard Result

     ![alt text](pictures/Imran_tensorboard_epochloss.png)
     ![alt text](pictures/Imran_tensorboard_epochmse.png)

## Results:

### Actual vs Predicted Covid-19 Cases

![alt text](pictures/Imran_predicted_vs_actual.png)

### Model Reports

![alt text](pictures/Imran_mse_mape_r2score.png)

## Build With:

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Acknowledgment of Data

The dataset used in this project was obtained from:
[Ministry of Health Malaysia's GitHub Repo](https://github.com/MoH-Malaysia/covid19-public).
