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
from Data_Organiser import Data_LSTM
from LSTM_Model import LSTM_Model
from Unit_Creation import Up_Down
from Purchase_Agent import Purchase_Agent
from Random_Purchase import Random_Purchase


'Is the main run file of LSTM, the settings for changing said model also exist here for ease of access'
def LSTM():
    #Settings
    epochs = 50
    batch_size = 1
    optimizer = 'adamax'
    loss='mean_squared_error'
    dropout_rate = 0.15
    settings = [epochs,batch_size,optimizer, loss, dropout_rate]
    
    #Data
    testing, company, x_train, y_train, x_test, scaler = Data_LSTM()
    
    #ANN Model
    prediction = LSTM_Model(settings, x_train, y_train, x_test, scaler)
    
    #Output
    Model_Output(testing['Prices'],company,prediction)

'Is simply the output of the model, containing classification and regression statisitcally information, formatted neatly for the user'
def Model_Output(prices,company,prediction):
    
    prices = (prices.dropna()).values
    #Units - Regression
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
    plt.title(f"LSTM Predictor Model")
    plt.ylabel(f"{company} - US Dollars")
    plt.xlabel(f"Time - Days")
    plt.legend()
    plt.show()
    return 0

LSTM()