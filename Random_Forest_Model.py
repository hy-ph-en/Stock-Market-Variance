from tkinter.tix import AUTO
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
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from Data_Organiser import Data_Random_Forest
from Unit_Creation import Standard_Deviation, Up_Down, RSI, Stochastic_Oscillator, Williams, MACD, Rate_Of_Change, Balance_Volume


'This function is what is essentially the model for random forest, put here as to simplify the main file "Random_Forest" it aids in running the code correctly and is'
'one of the more complex pieces of code in this project'
def RandomForest(n_estimators,oob_score,prediction_days,criterion,min_samples_leaf, training, testing):

    #Combing Dropping Nan and Combing Datasets
    nan_total_dataset = (pd.concat((training, testing), axis=0))
    total_dataset = (nan_total_dataset.copy()).dropna()
    total_dataset_index = total_dataset.index
    
    training = training.dropna()
    testing = testing.dropna()
    
    #Building Model
    random_forest = RandomForestRegressor(n_estimators = n_estimators, max_features='auto', min_samples_leaf=min_samples_leaf, max_depth=60, random_state = 0)
    
    #Model Testing
    overall_prediction = []
    length_difference = len(nan_total_dataset) - len(total_dataset) -1 
    num_days = len(testing.index)

    #Looping through the testing days 
    for x in range(0,num_days,prediction_days):
        
        #Get the most recent period of data
        currentSet = training.copy()
        skipDays = testing.iloc[:(x),:]
        currentSet = pd.concat([currentSet, skipDays])
        currentSet = (currentSet.iloc[(x):,:]).dropna()
        
        #Looping through the prediction days 
        for y in range(0,prediction_days):
            #Catch to stop dataset runover
            if(x+y>num_days-1):
                break
            
            price_data = 'Prices'
            predDay = testing.iloc[[(x+y)]]
            predicition_day = len(training)+x+y

            #Rebuilding the model to refit the new information
            random_forest.fit(currentSet[['RSI','Stochastic_Oscillator', 'r_percent','MACD','Rate_of_Change','balance_volume']],currentSet['Prices'])
            
            #Predicting current Day
            prediction = random_forest.predict(predDay[['RSI','Stochastic_Oscillator', 'r_percent','MACD','Rate_of_Change','balance_volume']])
            
            #Appending Prediction 
            overall_prediction.append(prediction[0])
        
            #Updating Variables to match found Price
            nan_total_dataset['Prices'] = nan_total_dataset['Prices'].replace(nan_total_dataset['Prices'].iloc[predicition_day+length_difference], prediction[0])
            nan_total_dataset['change_in_price'] = nan_total_dataset['Prices'].diff()
            
            standard_deviation = Standard_Deviation(nan_total_dataset['Prices'].iloc[(predicition_day+length_difference-14):(predicition_day+length_difference)])
            
            nan_total_dataset['Low'] = nan_total_dataset['Low'].replace(nan_total_dataset['Low'].iloc[predicition_day+length_difference], (nan_total_dataset['Prices'].iloc[predicition_day+length_difference] -standard_deviation))
            nan_total_dataset['High'] = nan_total_dataset['High'].replace(nan_total_dataset['High'].iloc[predicition_day+length_difference], (nan_total_dataset['Prices'].iloc[predicition_day+length_difference] +standard_deviation))
            
            new_dataset = nan_total_dataset[price_data].copy().values
            new_dataset = pd.DataFrame(new_dataset)
            new_dataset.columns = ['Prices']
            new_dataset['change_in_price'] = new_dataset.diff().values
            new_dataset['Low'] = nan_total_dataset['Low'].copy().values
            new_dataset['High'] = nan_total_dataset['High'].copy().values
            new_dataset['RSI'] = RSI(new_dataset)
            new_dataset['Stochastic_Oscillator'] = Stochastic_Oscillator(new_dataset, price_data).values
            new_dataset['Up_or_Down'] = Up_Down(nan_total_dataset['Prices'])
            new_dataset['r_percent'] = Williams(nan_total_dataset,price_data).values
            macd, macd_ema = MACD(nan_total_dataset,price_data)
            new_dataset['MACD'] = macd
            new_dataset['MACD_EMA'] = macd_ema
            new_dataset['Rate_of_Change'] = Rate_Of_Change(nan_total_dataset,price_data).values
            new_dataset['balance_volume'] = Balance_Volume(nan_total_dataset,price_data).values
            new_dataset.index = nan_total_dataset.index
            
            #Transfering Updated Indicators 
            newDays = testing.iloc[[(x+y)]]
            newIndex = newDays.index
            newDays.at[newIndex[0],'Prices'] = prediction[0]
            newDays.at[newIndex[0],'change_in_price'] = new_dataset.at[newIndex[0], 'change_in_price']
            newDays.at[newIndex[0],'Low'] = new_dataset.at[newIndex[0], 'Low']
            newDays.at[newIndex[0],'High'] = new_dataset.at[newIndex[0], 'High']
            newDays.at[newIndex[0],'RSI'] = new_dataset.at[newIndex[0], 'RSI']
            newDays.at[newIndex[0],'Stochastic_Oscillator'] = new_dataset.at[newIndex[0], 'Stochastic_Oscillator']
            newDays.at[newIndex[0],'r_percent'] = new_dataset.at[newIndex[0], 'r_percent']
            newDays.at[newIndex[0],'MACD'] = new_dataset.at[newIndex[0], 'MACD']
            newDays.at[newIndex[0],'Rate_of_Change'] = new_dataset.at[newIndex[0], 'Rate_of_Change']
            newDays.at[newIndex[0],'balance_volume'] = new_dataset.at[newIndex[0], 'balance_volume']
            
            currentSet = pd.concat([currentSet, newDays]).dropna()
            new_dataset = None
            
    #Edge case, add same price
    while(len(overall_prediction)<num_days):
        previousPrice = overall_prediction[len(overall_prediction)-1]
        overall_prediction.append(previousPrice)
    
    #Returning Prediction
    return overall_prediction
