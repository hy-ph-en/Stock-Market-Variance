import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import requests
import math

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from Data_Organiser import Data_Random_Forest
from Unit_Creation import Up_Down
from Random_Forest_Model import RandomForest
from Purchase_Agent import Purchase_Agent
from Random_Purchase import Random_Purchase

'Is the main run file of Random Forest, the settings for changing said model also exist here for ease of access'
def Random_Forest_Model():
    #Settings
    n_estimators = 500
    #600
    #400
    oob_score = True 
    criterion = 'squared_error'
    min_samples_leaf = 2
    
    #Data
    training, testing, company, prediction_days = Data_Random_Forest()
    
    #Model
    prediction = RandomForest(n_estimators,oob_score,prediction_days,criterion,min_samples_leaf,training,testing)

    #Model Output
    Model_Output(testing['Prices'], company, prediction)
    
    return 0
    
'Is simply the output of the model, containing classification and regression statisitcally information, formatted neatly for the user'
def Model_Output(prices,company,prediction):
    prices = (prices.dropna()).values
    #Units - Regression
    
    #preprocessing.normalize(a)
    r2 = r2_score(prices,prediction)
    mae = mean_absolute_error(prices,prediction)
    mse = mean_squared_error(prices,prediction)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs(prediction - prices)/np.abs(prices))
    
    report_reg = pd.DataFrame([r2,mae,mse,rmse,mape], index=['R2','Mean Absolute Error','Mean Squared Error','Root Mean Squared Error','Mean Absolute Percentage Error'], columns=['Output for Random Forest'])
    
    #Units - Classification 
    real_binary= Up_Down(prices)
    pred_binary= Up_Down(prediction)
    
    report_clt = pd.DataFrame(classification_report(y_true = real_binary, y_pred = pred_binary, target_names = ['Down', 'Up'], output_dict = True )).transpose()
    accuracy = accuracy_score(real_binary,pred_binary, normalize = True)*100
    predicted_gained = Purchase_Agent(prediction,prices)
    randomed_gained = Purchase_Agent(Random_Purchase(prediction),prices)
    
    #Output
    print("\33[4m" + "Regression Statistics")
    print(report_reg)
    print('')
    print("Classification Statistics")
    print(report_clt)
    print('')
    print("The Accuracy of the Model: %",accuracy)
    print("Predicted Purchase Money: £",predicted_gained)
    print("Random Purchase Money: £",randomed_gained)

    #Graphing
    plt.plot(prices, color = "black", label=f"Actual {company} Price")
    plt.plot(prediction, color="red", label=f"Predicted {company} Price")
    plt.title(f"Random Forest Predictor Model")
    plt.ylabel(f"{company} - US Dollars")
    plt.xlabel(f"Time - Days")
    plt.legend()
    plt.show()
    return 0

Random_Forest_Model()