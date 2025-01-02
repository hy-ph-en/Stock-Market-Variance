from operator import concat
import random
from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import math
from sklearn.preprocessing import MinMaxScaler

'An extremely simple function to see how a random purchase would act in each environment'
def Random_Purchase(data):
    buy_or_not = []
    for x in range(len(data)):
        buy_or_not.append(random.randint(0, 1))
        
    return buy_or_not