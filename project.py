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


def predict(model, X_test, pathname, order_no_test):
    model.eval()
    predictions = []

    for ex in X_test:
        pred = model.forward(torch.tensor(ex))
        predictions.append(pred.detach().numpy()[0])

    order_no_test['Time from Pickup to Arrival'] = pd.Series(predictions)
    order_no_test.set_index('Order No', inplace=True)
    order_no_test.to_csv('predictions.csv')
    return



def main():  
    X_train, y_train, X_test, order_no_test = prepForModel('data/Train.csv', 
                                            'data/Test.csv', 'data/Riders.csv') 

    trainNet(torch.tensor(X_train), torch.tensor(y_train.flatten()))
    trained_net = RegNet()
    trained_net.load_state_dict(torch.load('saved_model.pth'))
    
    predict(trained_net, X_test, 'predictions.csv', order_no_test)
    return


if __name__ == "__main__":
    main()

