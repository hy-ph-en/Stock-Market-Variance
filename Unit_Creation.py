from operator import concat
from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import math
from sklearn.preprocessing import MinMaxScaler

#Whether the day is up or down, in terms of price, compared to the previous day
def Up_Down(prices):
    prices_panda = pd.DataFrame(prices)
    up_or_down = prices_panda.transform(lambda x: x.shift(1) < x)
    up_or_down = up_or_down * 1
    return up_or_down

#Calculating the variance
def Variance(prices):
    length = len(prices)
    mean = sum(prices)/length
    deviation = [(price - mean) ** 2 for price in prices]
    variance = sum(deviation)/length
    return variance

#Calculating the standard deviation
def Standard_Deviation(prices):
    variance = Variance(prices)
    standard_deviation = math.sqrt(variance)
    return standard_deviation

#Caculating the relative strength index 
def RSI(data):
    #How Many Days to look back on - standard 
    days = 14
    
    #Getting the change in price, if the change is less than 0 set to 0
    up_data = data[['Prices','change_in_price']].copy()
    up_data.loc['change_in_price'] = up_data.loc[(up_data['change_in_price'] < 0), 'change_in_price'] = 0
    
    
    #Getting the change in price, if the change is more than 0 set to 0
    down_data = data[['Prices','change_in_price']].copy()
    down_data.loc['change_in_price'] = down_data.loc[(down_data['change_in_price'] > 0), 'change_in_price'] = 0
    
    #Get the absolute value, no negatives 
    down_data['change_in_price'] = down_data['change_in_price'].abs()
    
    
    #Calculating the Exponential Weighted Moving Average, EWMA
    ewm_up = up_data['change_in_price'].transform(lambda x: x.ewm(span = days).mean())
    ewm_down = down_data['change_in_price'].transform(lambda x: x.ewm(span = days).mean())
    
    #Calculating nominal strength
    strength_index = 100.0 - (100.0/(1.0 + ewm_up/ewm_down ))
    
    #Making Panda
    strength_index = pd.DataFrame(strength_index)
    strength_index.columns = ['RSI']
    
    #Returning created data frame
    return strength_index

#Calculating Stochastic Oscillator
def Stochastic_Oscillator(data, price_data):
    #How Many Days to look back on - standard 
    days = 14
    
    low_data, high_data = data[[price_data,'Low']].copy(), data[[price_data,'High']].copy()
    
    #Applying rolling function for Min and Max
    low = low_data['Low'].transform(lambda x: x.rolling(window = days).min())
    high = high_data['High'].transform(lambda x: x.rolling(window = days).max())
    
    #Calculating the Stochastic Oscillator
    k_percent = 100 * ((data[price_data] - low)/(high - low))
    
    #Making Panda
    k_percent = pd.DataFrame(k_percent)
    k_percent.columns = ['k_percent']
    
    #Returning created data frame
    return k_percent

#Calculating Williams
def Williams(data, price_data):
    #How Many Days to look back on - standard 
    days = 14
    
    low_data, high_data = data['Low'].copy(), data['High'].copy()
    
    #Appplying rolling function for Min and Max
    low = low_data.transform(lambda x: x.rolling(window = days).min())
    high = high_data.transform(lambda x: x.rolling(window = days).max())
    
    #Calculating Williams 
    r_percent = ((high - data[price_data])/(high - low)) * -100
    
    #Making Panda
    r_percent = pd.DataFrame(r_percent)
    r_percent.columns = ['r_percent']
    
    #Returning created data frame
    return r_percent


#Calculating MACD
def MACD(data, price_data):
    data_26, data_12 = data[price_data].copy(), data[price_data].copy()
    
    #Calculate MACD
    ema_26 = data_26.transform(lambda x: x.ewm(span = 26).mean())
    ema_12 = data_12.transform(lambda x: x.ewm(span = 12).mean())
    macd = ema_12 - ema_26
    
    #Calculate EMA - 9 is considered the signal line
    ema_9_macd = macd.ewm(span = 9).mean()
    
    #Creating Panda
    macd = pd.DataFrame(macd)
    macd.columns = ['MACD']
    
    ema_9_macd = pd.DataFrame(ema_9_macd) 
    ema_9_macd.columns = ['MACD_EMA']

    #Returning created data frame
    return macd, ema_9_macd

#Calculating Rate of Change
def Rate_Of_Change(data, price_data):
    #9 is the singal line again here
    span = 9
    
    data_close = data[price_data].copy()
    
    #Calculating rate of change
    rate_of_change = data_close.transform(lambda x: x.pct_change(periods = span))
    
    #Creating Panda
    rate_of_change = pd.DataFrame(rate_of_change)
    rate_of_change.columns = ['Rate_of_Change']
    
    #Returning created data frame
    return rate_of_change

#Calculating Balance Volume
def Balance_Volume(data, price_data):
    #Initalising variables
    volume = data['Volume'].copy()
    change = data[price_data].diff()

    prev_obv = 0
    obv_list = []

    #Calculate the On Balance Volume
    for x, y in zip(change, volume):
        if x > 0:
            current_obv = prev_obv + y
        elif x < 0:
            current_obv = prev_obv - y
        else:
            current_obv = prev_obv

        #Creating obv list
        prev_obv = current_obv
        obv_list.append(current_obv)
    
    #Returning created data frame
    return pd.Series(obv_list, index = data.index)