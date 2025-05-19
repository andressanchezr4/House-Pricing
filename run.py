# -*- coding: utf-8 -*-
"""
@author: andres.sanchez
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf

from DLs import Autoencoder, LossThresholdCallback, lr_schedule, plot_training_loss
from MLs import MLsPipeline
from Superlearner import SuperLearner
from EDA import EDA, impute_data

path = './data/'
df_train = pd.read_csv(path + 'train.csv')
df_train_noid = df_train.drop('Id', axis=1) 

df_test = pd.read_csv(path + 'test.csv')
df_test_noid = df_test.drop('Id', axis=1) 

nominal_columns = [
    'MSZoning', 'Street', 'Alley', 'LotConfig', 'Neighborhood', 'Condition1',
    'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
    'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
    'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'SaleType',
    'SaleCondition', 'MiscFeature'
]

ordinal_columns = [
    'LotShape', 'LandContour', 'Utilities', 'LandSlope', 'ExterQual',
    'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'GarageFinish',
    'GarageQual', 'GarageCond', 'FireplaceQu', 'Fence', 'PoolQC'
]

###########
### EDA ###
###########

# Explore and repare TRAIN DATA
path2figures = './figures/'
exploratory_analysis = EDA(df_train_noid, path2figures)

exploratory_analysis.general_overview()
exploratory_analysis.visualize_data()

# exploratory_analysis.high_correlation
# exploratory_analysis.corr_repare()

exploratory_analysis.NaN_analysis()
exploratory_analysis.columns2delete
df_train_nonan = exploratory_analysis.repare_nan()

# Explore and repare TEST DATA
path2figures = './figures_test/'
exploratory_analysis_test = EDA(df_test_noid, path2figures)

exploratory_analysis_test.general_overview()
exploratory_analysis_test.visualize_data()

# exploratory_analysis_test.high_correlation
# exploratory_analysis_test.corr_repare()

exploratory_analysis_test.NaN_analysis()
exploratory_analysis_test.columns2delete
df_test_nonan = exploratory_analysis_test.repare_nan()

# Data is joined for proper category level vectorization
common_columns = df_train_nonan.columns.intersection(df_test_nonan.columns)

# The sale price is saved to be added to the train set for proper imputation
df_train_sale_price = df_train_nonan.SalePrice

df_test_nonan = df_test_nonan[common_columns]
df_train_nonan = df_train_nonan[common_columns]

size_train = len(df_train_nonan)

df2impute = pd.concat([df_train_nonan, df_test_nonan])


########################
### Data Preparation ###
########################
nominal_columns = [col for col in nominal_columns if col in df2impute.columns]
ordinal_columns = [col for col in ordinal_columns if col in df2impute.columns]

# DISCLAIMER: Imputation does not take information from test and train
# together to impute data, as that would corrupt the training data
# (data leakage) making it easier for the model to make predictions on the test set.
# We only join them to simplify the process of vectorising the categorical variables.
df_train_imputed, df_test_imputed = impute_data(df2impute, 
                                                nominal_columns, 
                                                ordinal_columns, 
                                                df_train_sale_price, 
                                                n_neighbors=10, 
                                                joined = True, 
                                                size_split = size_train)

# TRAIN DATA preparation
df_train_imp, df_val_imp = train_test_split(df_train_imputed, test_size=0.1, random_state=1234)

train_y, train_x = df_train_imp[['SalePrice']], df_train_imp.drop(['SalePrice'], axis=1) 
val_y, val_x = df_val_imp[['SalePrice']], df_val_imp.drop(['SalePrice'], axis=1)

###############################
### Machine learning Models ###
###############################
path2scatterplots = './ml_predictions/'
pipeline = MLsPipeline(train_x, train_y, val_x, val_y, path2scatterplots)

# ---- Linear models ----
# Linear regression
lm_model, rmse = pipeline.fit_linear_models(model_type="linear")
lm_lasso, rmse_lasso = pipeline.fit_linear_models(use_lasso=True, model_type="linear")
print(f"Linear Regression RMSE: {rmse:.3f}")
print(f"Linear Regression RMSE: {rmse_lasso:.3f}") 

# Ridge regression
lm_ridge, rmse_ridge = pipeline.fit_linear_models(model_type="ridge")
lm_ridge_lasso, rmse_ridge_lasso = pipeline.fit_linear_models(use_lasso=True, model_type="ridge")
print(f"Ridge Regression RMSE: {rmse_ridge:.3f}") 
print(f"Ridge Regression RMSE + Lasso: {rmse_ridge_lasso:.3f}") 

# Elastic Net regression
lm_elne, rmse_elne = pipeline.fit_linear_models(model_type="elastic_net")
lm_elne_lasso, rmse_elne_lasso = pipeline.fit_linear_models(use_lasso=True, model_type="elastic_net")
print(f"Elastic Net Regression RMSE: {rmse_elne:.3f}")
print(f"Elastic Net Regression RMSE + Lasso: {rmse_elne_lasso:.3f}") 


# ---- Random Forrest ----
n_trees = [10, 50, 100, 200, 300, 500, 1000]
best_rf_model, best_rf_rmse = pipeline.fit_random_forest(n_trees)
best_rf_model_lasso, best_rf_rmse_lasso = pipeline.fit_random_forest(n_trees, use_lasso=True)
print(f"Best Random Forest RMSE: {best_rf_rmse:.3f}")
print(f"Best Random Forest RMSE + Lasso: {best_rf_rmse_lasso:.3f}") 


# ----- SVM -----
svm_model, rmse_svm = pipeline.fit_svm()
svm_model_lasso, rmse_svm_lasso = pipeline.fit_svm(use_lasso=True)
print(f"SVM RMSE: {rmse_svm:.3f}")
print(f"SVM RMSE + Lasso: {rmse_svm_lasso:.3f}") 


# ----- XGBoost ----- | BEST RESULTS IN TEST SET: 0.14 RMSE According to Kaggle
n_estimators = [100, 300, 500, 700, 900, 1200]
xgb_model, rmse_xgb = pipeline.fit_xgboost(n_estimators)
xgb_model_lasso, rmse_xgb_lasso = pipeline.fit_xgboost(n_estimators, use_lasso = True)
print(f"XGBoost RMSE: {rmse_xgb:.3f}") 
print(f"XGBoost RMSE + Lasso: {rmse_xgb_lasso:.3f}")

print(pipeline.results_df)
pipeline.results_df.to_csv(path2scatterplots +'results_mls.csv', index = None)

################################
### Stacked ML: SuperLearner ###
################################
my_sl = SuperLearner(train_x, train_y, path2scatterplots)
my_sl.fit()
y_pred_val = my_sl.predict(val_x)
print(f'SuperLearner RMSE: {np.sqrt(mean_squared_error(val_y, y_pred_val))}')
my_sl.evaluate_models(val_x, val_y) 
my_sl.plot_results(val_y, y_pred_val)

best_models = my_sl.best_models
optimised_sl = SuperLearner(train_x, train_y, path2scatterplots, model_list = best_models)
optimised_sl.fit()
y_pred_val_opt = optimised_sl.predict(val_x)
print(f'SuperLearner Optimised RMSE: {np.sqrt(mean_squared_error(val_y, y_pred_val_opt))}')
optimised_sl.evaluate_models(val_x, val_y)

#######################
### DL: Autoencoder ###
#######################
autoencoder = Autoencoder(train_x)

loss_threshold_callback = LossThresholdCallback(threshold=200)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
autoencoder.compile(optimizer=optimizer)

history = autoencoder.fit(train_x, train_x, epochs=100, 
                          batch_size=128, validation_data=(val_x, val_x),
                          callbacks=loss_threshold_callback
                          )

plot_training_loss(history.history)
# autoencoder.plot_latent_space(train_x, train_y.to_numpy().ravel())

decoded_data, latent_space = autoencoder.predict(train_x)
decoded_data_val, latent_space_val = autoencoder.predict(val_x)

# We use the autoencoder latent space to train another model 
my_sl = SuperLearner(latent_space, train_y, path2scatterplots)
my_sl.fit()
y_pred_val = my_sl.predict(latent_space_val)
print(f'SuperLearner RMSE: {np.sqrt(mean_squared_error(val_y, y_pred_val))}')

##################
### Submission ###
##################
# We train the best model with the whole data and make predictions on the test set.
xgb_model.fit(df_train_imputed.drop(['SalePrice'], axis = 1), df_train_imputed[['SalePrice']])
y_pred_test = xgb_model.predict(df_test_imputed)
df2send = pd.DataFrame([df_test['Id'].values, y_pred_test.flatten()]).transpose()
df2send.columns = ['Id', 'SalePrice']
df2send.Id = df2send.Id.astype('int64')
df2send.to_csv('./submission.csv', index = None) 





