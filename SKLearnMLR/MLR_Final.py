#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 00:02:46 2023

@author: loganbartholomew
"""

# Author: George Logan Bartholomew
# Date Finalized: August 3rd, 2023
# Email: logan_bartholomew@berkeley.edu

'''
This program performs K-Fold cross-validation and multivariable linear 
regression analysis on molecular parameters held in an Excel
spreadsheet.
'''

# changes the working directory to the directory this script is located in
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score, KFold

# Input arrays
worksheet = pd.read_excel('molecular_parameters_total.xlsx', header=None)
array = np.array(worksheet)

# Parameter names
parameter_names = array[0,1:-1]

# Compound names
compound_names = array[1:,0]

# Parameter data
params = array[1:,1:-2]
params = params.astype(np.float64)

# Yield data
reaction_yields = np.ndarray.tolist(array[1:,-1])

# List of MLR models to use
mlr_models = [
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge()),
    ("Lasso Regression", Lasso())
]

# Perform feature selection to find the best combination of n parameters
k_best = 3  # Number of parameters to select
selector = SelectKBest(score_func=f_regression, k=k_best)
selected_params = selector.fit_transform(params, reaction_yields)
selected_param_indices = selector.get_support(indices=True)
selected_parameter_names = [parameter_names[i] for i in selected_param_indices]

for model_name, model in mlr_models:
    print(f"Using {model_name}:")
    
    # Perform multivariate linear regression with K-fold cross-validation using the selected parameters
    cv = KFold(n_splits=4, 
               shuffle=True, 
               random_state=42)
    
    cross_val_scores = cross_val_score(model, 
                                       selected_params, 
                                       reaction_yields, 
                                       scoring='r2', 
                                       cv=cv)
    
    # Print the cross-validated R-squared scores
    print("Cross-validated R-squared scores:", cross_val_scores)
    
    # Calculate the coefficients and intercept
    model.fit(selected_params, reaction_yields)
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Print the equation
    equation = f'Y = {intercept:.2f}'
    for i, coef in zip(selected_param_indices, coefficients):
        equation += f' + {coef:.2f} * {parameter_names[i]}'
    
    print("Equation:", equation)
    
    # Plot the model as a scatter plot with line of best fit
    predicted_yields = model.predict(selected_params)
    
    # Calculate correlation coefficients between all parameters and reaction yields
    all_correlation_scores = np.corrcoef(params, reaction_yields, rowvar=False)[-1, :-1]
    
    ''' Uncomment to print out individual correlation scores
    # Step 12: Print individual correlation scores for all parameters
    print("Individual correlation scores:")
    for parameter_name, corr_score in zip(parameter_names, all_correlation_scores):
        print(f"{parameter_name}: {corr_score:.2f}")
    '''
    
    # Calculate individual R-squared and stddev data for models
    cross_val_scores = cross_val_score(model, 
                                       selected_params, 
                                       reaction_yields, 
                                       scoring='r2', 
                                       cv=cv)
    
    # Print R-squared scores
    print("Cross-validated R-squared scores:", cross_val_scores)
    print(f"Mean R-squared score: {np.mean(cross_val_scores):.2f}")
    print(f"Standard deviation of R-squared scores: {np.std(cross_val_scores):.2f}")
    
    
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of predicted yields
    plt.scatter(reaction_yields, predicted_yields, color='darkcyan')
    
    # Line of best fit
    plt.plot([min(reaction_yields), max(reaction_yields)], [min(predicted_yields), max(predicted_yields)], color='darkslategrey', linestyle='--')
    
    plt.xlabel('Actual Yields', fontsize=12, fontname='Avenir')
    plt.ylabel('Predicted Yields', fontsize=12, fontname='Avenir')
    plt.title(f'Predicted vs. Actual Yields: All Substrates ({model_name})', fontsize=14, fontname='Avenir')
    plt.xticks(fontname='Avenir')
    plt.yticks(fontname='Avenir')
    plt.tight_layout()
    
    # Increase plot resolution (DPI)
    plt.savefig(f'scatter_plot_{model_name.replace(" ", "_").lower()}.png', dpi=300)  # Adjust the filename and DPI as per your preference
    
    plt.show()
