from operator import index
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import math 

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from Data_Organiser import OrganisingData_LSTM
from Unit_Creation import Standard_Deviation, Up_Down, RSI, Stochastic_Oscillator, Williams, MACD, Rate_Of_Change, Balance_Volume

'This function contains the actual model for LSTM '
def LSTM_Model(settings, x_train, y_train, x_test, scaler):
    #Settings
    #0 - epochs
    #1 - batch_size
    #2 - optimizer
    #3 - loss
    #4 - dropout_rate
    
    #LSTM Model Creation 
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(Dropout(settings[4]))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(settings[4]))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(settings[4]))
    model.add(Dense(units=1))
    model.compile(optimizer=settings[2], loss=settings[3])
    #Model Training
    model.fit(x_train, y_train, epochs=settings[0], batch_size=settings[1])

    #Model Testing
    overall_prediction = model.predict(x_test)
    
    
    #Inversing Scalar
    overall_prediction = overall_prediction.reshape(overall_prediction.shape[0], (overall_prediction.shape[1]*overall_prediction.shape[2]))
    overall_prediction = np.repeat(overall_prediction, x_test.shape[2], axis=-1)
    overall_prediction = scaler.inverse_transform(overall_prediction)[:,0]

    #Returning Prediction
    return overall_prediction



