import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataProcessing import *
import missingno as msno

#missing values 
#feature engineering: rider tiers? keep original features after making new column?
#low corr for rider features
#code organization: make a wrapper function that does all the data processing

def main():  
    X_train, y_train, X_test, order_no_test = prepForModel('data/Train.csv', 'data/Test.csv', 'data/Riders.csv') 
    print(f"train: \n{X_train.shape}\ny:\n{y_train.head()}\ntest:\n{X_test.shape}\norderno:\n{order_no_test.head()}") 
    return


if __name__ == "__main__":
    main()

