# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:19:40 2024

@author: andres.sanchez
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
import os
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class MLsPipeline(object):
    
    def __init__(self, train_x, train_y, val_x, val_y, path2figures, random_state=1234):
        self.train_x = train_x
        self.val_x = val_x

        self.random_state = random_state
        
        self.train_y = train_y.to_numpy().ravel()
        self.val_y = val_y.to_numpy().ravel()
        
        self.results_df = pd.DataFrame(columns=["Model", "Use LASSO", "RMSE"])
        
        self.path2figures = path2figures
        if not os.path.exists(self.path2figures):
            os.mkdir(self.path2figures)
            
    def lasso_feature_selection(self, n_cv):
        lasso = LassoCV(cv = n_cv, tol=1e-4, random_state = self.random_state)
        lasso.fit(self.train_x, self.train_y)
        self.selected_features = self.train_x.columns[lasso.coef_ != 0]
        
        train_x_reduced = self.train_x[self.selected_features]
        val_x_reduced = self.val_x[self.selected_features]
        
        return train_x_reduced, val_x_reduced
    
    def plot_results(self, y, y_pred, model_used, use_lasso, color_plot):
        if use_lasso:
            lasso = 'Lasso_reduced'
        else:
            lasso = '(no_Lasso)'

        coefficients = np.polyfit(y_pred, y, 1)  
        regression_line = np.polyval(coefficients, y_pred)  
        
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, y, color=color_plot, alpha=0.7, edgecolors='k')
        plt.plot(y_pred, regression_line, color='red', label=f'Regression Line (y={coefficients[0]:.2f}x + {coefficients[1]:.2f})')
        plt.title(f'Scatter Plot for {model_used} {lasso}')
        plt.xlabel('Predicted_values')
        plt.ylabel('Real_values')
        plt.grid(True)
        plt.savefig(self.path2figures + f"{model_used}_{lasso}_scatter_plot.png", dpi=300, 
                    bbox_inches='tight')
        plt.show()
        
        mse = (np.array(y) - np.array(y_pred))**2
        sorted_mse = np.sort(mse)
        yvals = np.linspace(0, 1, len(sorted_mse), endpoint=False)
        plt.plot(sorted_mse, yvals, marker='.', linestyle='-', color = color_plot)
        plt.title(f'Accumulated MSE {model_used} {lasso}')
        plt.xlabel('MSE')
        plt.ylabel('Fraction of samples')
        plt.grid(True)
        plt.savefig(self.path2figures + "{model_used}_{use_lasso}_accumulated_mse.png", dpi=300, 
                    bbox_inches='tight')
        plt.show() 
    
    def update_rmse_df(self, model_name, use_lasso, rmse):
        # As the results_df dataframe is not going to be too big
        # we can simply grow it
        self.results_df.loc[len(self.results_df)] = model_name, use_lasso, rmse
    
    def fit_random_forest(self, n_trees, use_lasso = False, n_cv = 5):
        if use_lasso:
            train_x_reduced, val_x_reduced = self.lasso_feature_selection(n_cv)
            
        else:
            train_x_reduced = self.train_x
            val_x_reduced = self.val_x
            
        mean_scores = []
        if isinstance(n_trees, list):
            
            for n in n_trees:
                rf = RandomForestRegressor(n_estimators=n, random_state=self.random_state)
                scores = cross_val_score(rf, train_x_reduced, self.train_y, 
                                         cv=n_cv, scoring="neg_mean_squared_error")
                mean_scores.append(-np.mean(scores))
                print(f'trying with {n} trees... Mean score: {-np.mean(scores):.3f}')
    
            plt.figure(figsize=(8, 5))
            plt.plot(n_trees, mean_scores, marker='o', linestyle='-')
            if use_lasso:
                plt.title(f"Number of Trees vs Cross-Validated RMSE with {train_x_reduced.shape[1]} variables")
            else:
                plt.title("Number of Trees vs Cross-Validated RMSE")
            plt.xlabel("Number of Trees")
            plt.ylabel("RMSE")
            plt.grid()
            plt.show()

            n_sco = float('inf') 
            self.n_tree = None  
            for sco, tree in zip(mean_scores, n_trees):
                if sco < n_sco:  
                    n_sco = sco
                    self.n_tree = tree

            rf = RandomForestRegressor(n_estimators=self.n_tree, random_state=self.random_state)

        elif isinstance(n_trees, (float, int)):
            
            print(f'Using {int(n_trees)}')
            rf = RandomForestRegressor(n_estimators=int(n_trees), random_state=self.random_state)
        
        else:
            ValueError("Unsupported format: n_trees")
        
        rf.fit(train_x_reduced, self.train_y)
        y_pred = rf.predict(val_x_reduced)
        rmse = np.sqrt(mean_squared_error(self.val_y, y_pred))
        
        self.plot_results(y_pred, self.val_y, 'RF', use_lasso, 'green')
        
        self.update_rmse_df("Random Forest", use_lasso, rmse)
        
        return rf, rmse

    def fit_linear_models(self, n_cv = 5, use_lasso=False, model_type="linear"):
        if use_lasso:
            train_x_reduced, val_x_reduced = self.lasso_feature_selection(n_cv)
            
        else:
            train_x_reduced = self.train_x
            val_x_reduced = self.val_x
        
        if model_type == "linear":
            lm_model = LinearRegression()
        
        elif model_type == "ridge":
            lm_model = RidgeCV(alphas=np.logspace(-10, 1, 100), cv=n_cv)
        
        elif model_type == "elastic_net":
            lm_model = ElasticNetCV(l1_ratio=np.linspace(0.1, 1.0, 10), alphas=np.logspace(-10, 1, 100),
                                 cv=n_cv,
                                 random_state=self.random_state)
        else:
            raise ValueError("Unsupported model type")
        
        lm_model.fit(train_x_reduced, self.train_y)
        y_pred = lm_model.predict(val_x_reduced)
        rmse = np.sqrt(mean_squared_error(self.val_y, y_pred))
        
        self.update_rmse_df(model_type.capitalize(), use_lasso, rmse)
        
        self.plot_results(self.val_y, y_pred, model_type, use_lasso, 'black')
        
        return lm_model, rmse
    
    def fit_svm(self, n_cv = 5, use_lasso=False):
        if use_lasso:
            train_x_reduced, val_x_reduced = self.lasso_feature_selection(n_cv)
        
        else:
            train_x_reduced = self.train_x
            val_x_reduced = self.val_x
        
        svm_model = SVR()
        svm_model.fit(train_x_reduced, self.train_y)
        y_pred = svm_model.predict(val_x_reduced)
        rmse = np.sqrt(mean_squared_error(self.val_y, y_pred))
    
        self.plot_results(y_pred, self.val_y, 'SVM', use_lasso, 'purple')  
        
        self.update_rmse_df("SVM", use_lasso, rmse)
        
        return svm_model, rmse


    def fit_xgboost(self, n_estim, n_cv = 5, use_lasso = False):
        if use_lasso:
            train_x_reduced, val_x_reduced = self.lasso_feature_selection(n_cv)
        else:
            train_x_reduced = self.train_x
            val_x_reduced = self.val_x
        
        if isinstance(n_estim, list):
            
            mean_scores = []
            for n in n_estim:
                xgb_model = XGBRegressor(
                    random_state=self.random_state,
                    n_estimators=n,  
                    objective='reg:squarederror', 
                    eval_metric='rmse'
                )
                scores = cross_val_score(xgb_model, train_x_reduced, self.train_y, 
                                         cv=n_cv, scoring="neg_mean_squared_error")
                mean_scores.append(-np.mean(scores))
                print(f'trying with {n} estimators... Mean score: {-np.mean(scores):.3f}')
    
            plt.figure(figsize=(8, 5))
            plt.plot(n_estim, mean_scores, marker='o', linestyle='-')
            if use_lasso:
                plt.title(f"Number of Estimators vs Cross-Validated RMSE with {train_x_reduced.shape[1]} variables")
            else:
                plt.title("Number of Estimators vs Cross-Validated RMSE")
            plt.xlabel("Number of Estimators")
            plt.ylabel("RMSE")
            plt.grid()
            plt.show()

            n_sco = float('inf') 
            self.n_est = None  
            for sco, est in zip(mean_scores, n_estim):
                if sco < n_sco:  
                    n_sco = sco
                    self.n_est = est
            
            xgb_model = XGBRegressor(
                random_state=self.random_state,
                n_estimators=self.n_est, 
                objective='reg:squarederror',
                eval_metric='rmse'
            )
        
        elif isinstance(n_estim, (float, int)):   
            
            print(f'Build XGBoost with {n_estim}')
            xgb_model = XGBRegressor(
                random_state=self.random_state,
                n_estimators=int(n_estim), 
                objective='reg:squarederror',  
                eval_metric='rmse'
            )
        
        else:
            raise ValueError("Unsupported n_estimators type")
    
        xgb_model.fit(train_x_reduced, self.train_y)
        y_pred = xgb_model.predict(val_x_reduced)
        rmse = np.sqrt(mean_squared_error(self.val_y, y_pred))

        self.plot_results(y_pred, self.val_y, 'XGBoost', use_lasso, 'blue')        

        self.update_rmse_df("XGBoost", use_lasso, rmse)
    
        return xgb_model, rmse







