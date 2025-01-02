from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

'Purchase Agent shows how profitable a prediction model is'
def Purchase_Agent(predicted_data,real_data):      
    #Testing Parameters
    money = 5000
    units = 0
    
    #Loops throygh the real and predicted data to see effectiveness
    for x in range(len(real_data)-1):
        tomorrow_prediction_price = predicted_data[x+1]
        current_prediction_price = predicted_data[x]
        current_real_price = real_data[x]
        
        #Sell
        if(current_prediction_price >= tomorrow_prediction_price and units != 0):
            money = units * current_real_price   
        #Buy
        if(current_prediction_price < tomorrow_prediction_price and money != 0):
            units = current_real_price/money 
            
    #A catch to make sure its all converted to money
    if(units != 0 ):
        money = units * real_data[len(real_data) - 1]
        
    #Returns Money
    return money
