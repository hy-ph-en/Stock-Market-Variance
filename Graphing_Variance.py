from operator import concat
from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from Unit_Creation import Standard_Deviation
from Data_Organiser import Data_Random_Forest

'This function converts a list of prices into their variance'
def Graphing_Variance():
    #Inputs
    training, testing, company, prediction_days = Data_Random_Forest()
    total_data = pd.concat((training, testing), axis=0)
    
    #Settings
    prediction_days = 30
    
    #Variables
    standard_deviations = []
    iterations = round((len(total_data)/prediction_days))
    count = 0

    #Loops through the data in segments according to prediction days
    for x in range(iterations): 
        print(total_data['Prices'].iloc[(count):(count + prediction_days)])
        standard_deviations.append(Standard_Deviation(total_data['Prices'].iloc[(count):(count + prediction_days)]))
        count = count + prediction_days
    
    #Converets the Variance into a more useable 0-1 statistic
    scaled = MinMaxScaler(feature_range=(0,1))
    standard_deviations = np.array(standard_deviations)
    standard_deviations = scaled.fit_transform(standard_deviations.reshape(-1,1))

    #Outputs the found variance over time
    plt.plot(standard_deviations, color = "black", label=f"Actual {company} Variance")
    plt.title(f"Variance Display")
    plt.ylabel(f"{company} - Variance")
    plt.xlabel(f"Time - {prediction_days} Day Segment")
    plt.hlines(y = 0.5, xmin = 0, xmax = len(standard_deviations), color = 'r', linestyles = 'dashed')
    plt.hlines(y = 0.35, xmin = 0, xmax = len(standard_deviations), color = 'y', linestyles = 'dashed')
    plt.hlines(y = 0.3, xmin = 0, xmax = len(standard_deviations), color = 'g', linestyles = 'dashed')
    plt.legend()
    plt.show()
    
Graphing_Variance()