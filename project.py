import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataProcessing import *
import missingno as msno
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from model import *


#missing values 
#feature engineering: rider tiers? keep original features after making new column?
#low corr for rider features
#code organization: make a wrapper function that does all the data processing

def main():  
    X_train, y_train, X_test, order_no_test = prepForModel('data/Train.csv', 'data/Test.csv', 'data/Riders.csv') 
    #print(f"train: \n{X_train.shape}\ny:\n{y_train.head()}\ntest:\n{X_test.shape}\norderno:\n{order_no_test.head()}") 

    # XGBModel = XGBRegressor()
    # XGB_cv_scores = cross_val_score(XGBModel, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")

    # RFModel = RandomForestRegressor()
    # RF_cv_scores = cross_val_score(RFModel, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")


    # ADBModel = AdaBoostRegressor()
    # ADB_cv_scores = cross_val_score(ADBModel, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")

    # XGBModel = RandomForestRegressor()
    # XGBModel.fit(X_train, y_train)
    # XGBpredictions = XGBModel.predict(X_test)

    trained_net = trainNet(torch.tensor(X_train), torch.tensor(y_train.to_numpy().flatten()))
    predictions = []
    for ex in X_test:
        pred = trained_net.forward(torch.tensor(ex))
        predictions.append(pred.detach().numpy()[0])

    order_no_test['Time from Pickup to Arrival'] = pd.Series(predictions)

    
    
    # order_no_test['Time from Pickup to Arrival'] = pd.DataFrame(XGBpredictions)
    order_no_test.set_index('Order No', inplace=True)
    print(order_no_test.head())
    order_no_test.to_csv('predictions.csv')


    return


if __name__ == "__main__":
    main()

