# House Price Kaggle Competition
This repo contains an approach to find a solution for the competition from Kaggle: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview).

The best model (XGBoost) achieved a score of 0.14. When trained with the data embedding from the autoencoder, the predictions are slightly improved. 

According to [the analysis of fedesoriano](https://www.kaggle.com/code/fedesoriano/house-prices-what-s-a-good-score#4.2.-Model-fitting), a good and realistic model should be able to 
accomplish a score between 0.10 and 0.77, whilst a top model should score between 0.10 and 0.14.

## Script Description
* run.py --> End to end analysis.
* EDA.py --> Exploratory Data Analysis and data depuration.
* MLs.py --> Machine Learning implementations.
* Superlearner --> Stacked Machine Learning model implmementation.
* DLs.py --> Autoencoder Implementation.

## Requirements
* Numpy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* xgboost
* Tensorflow
