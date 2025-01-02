from operator import concat
from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import StandardScaler
from Unit_Creation import Up_Down, RSI, Stochastic_Oscillator, Williams, MACD, Rate_Of_Change, Balance_Volume

'This function allows the user to change the data settings for both LSTM and Random Forest Simultaneously, allowing for an easier user experince'
def Settings():
    #Settings 
    prediction_days = 1
    company = 'ETH-USD'
    price_data = 'Adj Close'
    
    #You would want the training dataset to bleed 14 days into the testing due to dropping 14 rows due to how the units are created#
    start_train = dt.datetime(2018, 1, 13)
    end_train = dt.datetime(2018, 6, 17)
    
    start_test = dt.datetime(2019, 6, 18)
    end_test = dt.datetime(2019, 7, 1)
    
    #check price against yahoo prices at the start 
    return prediction_days, company, price_data, start_train, end_train, start_test, end_test

'This function organises the data for LSTM'
def Data_LSTM():
    #Settings 
    prediction_days, company, price_data, start_train, end_train, start_test, end_test = Settings()

    #Loading the Data 
    data_train = pd.DataFrame(web.DataReader(company, 'yahoo', start_train, end_train))
    data_test  = pd.DataFrame(web.DataReader(company, 'yahoo', start_test, end_test))

    total_data = pd.concat((data_train, data_test), axis=0)
    
    #Real Prices
    total = total_data[price_data].values

    #Dataset Creation - Total 
    total = pd.DataFrame(total)
    total.columns = ['Prices']
    total['change_in_price'] = total.diff()
    total['RSI'] = RSI(total)
    total['Stochastic_Oscillator'] = Stochastic_Oscillator(total_data, price_data).values
    total['Up_or_Down'] = Up_Down(total['Prices'])
    total['r_percent'] = Williams(total_data, price_data).values
    macd, macd_ema = MACD(total_data, price_data)
    total['MACD'], total['MACD_EMA'] = macd.values, macd_ema.values
    total['Rate_of_Change'] = Rate_Of_Change(total_data, price_data).values
    total['balance_volume'] = Balance_Volume(total_data, price_data).values
    total['Low'] = total_data['Low'].values
    total['High'] = total_data['High'].values
    total['Volume'] = total_data['Volume'].values
    prediction_days = 1
    total.index = total_data.index

    #Training
    training = total.iloc[0:(len(data_train)-1)]
    
    #Testing
    testing = total.iloc[len(data_train):len(total)]
    
    #Making the Data into the Correct format for LSTM 
    x_train, y_train, x_test, scaler = OrganisingData_LSTM(training,testing,prediction_days)

    #Organised Data Return
    return testing, company, x_train, y_train, x_test, scaler

'This function helps make the data into the right organisational structure for LSTM'
def OrganisingData_LSTM(training,testing,prediction_days):
    #Variables
    x_train = []
    y_train = []
    scaler = StandardScaler()
    training = training.copy().dropna()
    testing = testing.copy().dropna()
    total_dataset = pd.concat((training, testing), axis=0)
    total_dataset = total_dataset[['Prices','RSI','Stochastic_Oscillator', 'r_percent','MACD','Rate_of_Change','balance_volume']].values
    scaler = scaler.fit(training[['Prices','RSI','Stochastic_Oscillator', 'r_percent','MACD','Rate_of_Change','balance_volume']].values)
    training = scaler.transform(training[['Prices','RSI','Stochastic_Oscillator', 'r_percent','MACD','Rate_of_Change','balance_volume']].values)
    
    #Training Data
    for x in range(prediction_days, len(training)):
        x_train.append(training[x-prediction_days:x])
        y_train.append(training[x,0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    #Creating Scalar
    test_dataset = total_dataset[len(total_dataset)-len(testing)-prediction_days:]
    test_dataset = scaler.transform(test_dataset)
        
    #Testings Data
    x_test=[]
    for x in range(prediction_days, len(test_dataset)):
        x_test.append(test_dataset[x-prediction_days:x])

    x_test = np.array(x_test)
    
    #Returning Modifed Data
    return x_train, y_train, x_test, scaler

'This function organises the data for Random Forest'
def Data_Random_Forest():
    #Settings 
    prediction_days, company, price_data, start_train, end_train, start_test, end_test = Settings()

    #Loading the Data 
    data_train = pd.DataFrame(web.DataReader(company, 'yahoo', start_train, end_train))
    data_test  = pd.DataFrame(web.DataReader(company, 'yahoo', start_test, end_test))

    total_data = pd.concat((data_train, data_test), axis=0)
    
    #Real Prices
    total = total_data[price_data].values

    #Dataset Creation - Total 
    total = pd.DataFrame(total)
    total.columns = ['Prices']
    total['change_in_price'] = total.diff()
    total['RSI'] = RSI(total)
    total['Stochastic_Oscillator'] = Stochastic_Oscillator(total_data, price_data).values
    total['Up_or_Down'] = Up_Down(total['Prices'])
    total['r_percent'] = Williams(total_data, price_data).values
    macd, macd_ema = MACD(total_data, price_data)
    total['MACD'], total['MACD_EMA'] = macd.values, macd_ema.values
    total['Rate_of_Change'] = Rate_Of_Change(total_data, price_data).values
    total['balance_volume'] = Balance_Volume(total_data, price_data).values
    total['Low'] = total_data['Low'].values
    total['High'] = total_data['High'].values
    total['Volume'] = total_data['Volume'].values
    total.index = total_data.index

    #Training
    training = total.iloc[0:(len(data_train)-1)]
    
    #Testing
    testing = total.iloc[len(data_train):len(total)]

    #Organised Data Return
    return training, testing, company, prediction_days

Data_LSTM()
