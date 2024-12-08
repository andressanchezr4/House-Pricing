# -*- coding: utf-8 -*-
"""
@author: andres.sanchez
"""
from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class SuperLearner(object):
    
    def __init__(self, train_x, train_y, path2figures, 
                 model_list = [], meta_model = LinearRegression(), 
                 ):
       
        self.train_y = train_y.to_numpy().ravel()
        if not isinstance(train_x, np.ndarray):
            self.train_x = train_x.to_numpy()
        else:
            self.train_x = train_x
        if np.issubdtype(self.train_y.dtype, np.number):
            self.regression = True
        else:
            self.regression = False
          
        self.meta_model = meta_model
        if len(model_list) == 0:
            self.get_base_models()
        else:
            self.base_models = model_list
       
        self.path2figures = path2figures
        self.n_cv = len(self.base_models)
        
        rows_per_part = len(self.train_x) // self.n_cv  # Compute rows per part
        self.train_x = self.train_x[:rows_per_part * self.n_cv]
        self.train_y = self.train_y[:rows_per_part * self.n_cv]
        
    def get_base_models(self, random_state=1234):
        
        if self.regression:
            self.base_models_raw = self.base_models = [
                DecisionTreeRegressor(random_state=random_state),
                SVR(gamma="scale"),
                KNeighborsRegressor(),
                AdaBoostRegressor(random_state=random_state),
                BaggingRegressor(n_estimators=200, random_state=random_state),
                RandomForestRegressor(n_estimators=300, random_state=random_state),
                ExtraTreesRegressor(n_estimators=200, random_state=random_state),
                XGBRegressor(
                    random_state=random_state,
                    n_estimators=100,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                            ),
                ]
        
        else:
            self.base_models_raw= self.base_models = [
                LogisticRegression(solver='liblinear'),
                DecisionTreeClassifier(random_state=random_state),
                SVC(gamma='scale', probability=True),
                GaussianNB(),
                KNeighborsClassifier(),
                AdaBoostClassifier(random_state=random_state),
                BaggingClassifier(n_estimators=200, random_state=random_state),
                RandomForestClassifier(n_estimators=300, random_state=random_state),
                ExtraTreesClassifier(n_estimators=200, random_state=random_state),
                XGBClassifier(
                    random_state=random_state,
                    n_estimators=100,
                    objective="binary:logistic",
                    eval_metric="logloss",
                              ),
                ]
    def plot_results(self, y, y_pred):
        
        coefficients = np.polyfit(y_pred, y, 1)  
        regression_line = np.polyval(coefficients, y_pred)  
        
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, y, color='royalblue', alpha=0.7, edgecolors='k')
        plt.plot(y_pred, regression_line, color='red') #=f'Regression Line (y={coefficients[0]:.2f}x + {coefficients[1]:.2f})'
        plt.title('Scatter Plot for SuperLearner')
        plt.xlabel('Predicted_values')
        plt.ylabel('Real_values')
        plt.grid(True)
        plt.savefig(self.path2figures + "SuperLearner_scatter_plot.png", dpi=300, 
                    bbox_inches='tight')
        plt.show() 
        
        mse = (np.array(y) - np.array(y_pred))**2
        sorted_mse = np.sort(mse)
        yvals = np.linspace(0, 1, len(sorted_mse), endpoint=False)
        plt.plot(sorted_mse, yvals, marker='.', linestyle='-', color = 'red')
        plt.title('SuperLearner Accumulated MSE')
        plt.xlabel('MSE')
        plt.ylabel('Fraction of samples')
        plt.savefig(self.path2figures + "SuperLearner_accumulated_mse.png", dpi=300, 
                    bbox_inches='tight')
        plt.show() 
        
    def train_each_model_on_fold(self):
        meta_X, meta_y = [], []
        
        kfold = KFold(n_splits=self.n_cv, shuffle=True)
        	
        for n, (train_ix, test_ix) in enumerate(kfold.split(self.train_x)):
            fold_yhats = []
            train_X, test_X = self.train_x[train_ix], self.train_x[test_ix]
            train_y, test_y = self.train_y[train_ix], self.train_y[test_ix]
            meta_y.extend(test_y)
        		
            for m, model in enumerate(self.base_models):
                model.fit(train_X, train_y)
                yhat = model.predict(test_X)
                fold_yhats.extend(yhat)
                print(f'Done with {model.__class__()}')
                print(f'{m+1}/{len(self.base_models)} models')
                print(f'-------------{n+1}/{self.n_cv} CVs-------------')
            
            meta_X.append(np.hstack(fold_yhats))

        self.oof_X = np.vstack(meta_X).transpose()
        self.oof_y = np.asarray(meta_y)
    
    def fit_meta_model(self):
        self.meta_model.fit(self.oof_X, self.oof_y)
               
    def fit_base_models(self):
        for model in self.base_models:
        		model.fit(self.train_x, self.train_y)
    
    def evaluate_models(self, val_x, val_y):
        all_rmse = {}
        print('Each model individual RMSE')
        print('-'*20)
        for model in self.base_models:
            y_pred = model.predict(val_x)
            mse = np.sqrt(mean_squared_error(val_y, y_pred))
            all_rmse[model] = mse
            print(f'{model.__class__.__name__} RMSE: {mse:.3f}')
        
        mean_rmse = np.mean(list(all_rmse.values()))
        std_rmse = np.std(list(all_rmse.values()))
        
        self.best_models = []
        for model, error in all_rmse.items():
            if error <= mean_rmse + std_rmse:
                self.best_models.append(model)
    
    def predict(self, X):
        
        meta_X = list()
        
        for model in self.base_models:
            y_pred = model.predict(X)
            meta_X.append(y_pred.reshape(len(y_pred),1))
      	
        meta_X = np.hstack(meta_X)
        
        return self.meta_model.predict(meta_X)
        
    def fit(self):
    
        self.train_each_model_on_fold()
        self.fit_base_models()
        self.fit_meta_model()
        
    
