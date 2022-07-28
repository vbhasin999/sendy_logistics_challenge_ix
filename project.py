import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataProcessing import *
import missingno as msno
from sklearn import metrics
from sklearn.linear_model import LinearRegression
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


def compare_models(X,y):
    RFReg = RandomForestRegressor()
    XGBReg = XGBRegressor()
    ABReg = AdaBoostRegressor()
    LinReg = LinearRegression()

    RF_score = -cross_val_score(RFReg, X=X, y=y, scoring='neg_root_mean_squared_error')
    XGB_score = -cross_val_score(XGBReg, X=X, y=y, scoring='neg_root_mean_squared_error')
    AB_score = -cross_val_score(ABReg, X=X, y=y, scoring='neg_root_mean_squared_error')
    LinReg_score = -cross_val_score(LinReg, X=X, y=y, scoring='neg_root_mean_squared_error')
    x = np.arange(5)

    means = [RF_score.mean(), XGB_score.mean(), AB_score.mean(), LinReg_score.mean()]
    print(means)
    names = ['RF', 'XGB', 'AB']

    comb_scores = np.hstack((RF_score.reshape((5,1)), XGB_score.reshape((5,1)), AB_score.reshape((5,1)), LinReg_score.reshape((5,1))))
    df = pd.DataFrame(comb_scores, columns=['RandomForest', 'XGBoost', 'AdaBoost', 'LinReg'])

    print(df.head())
    dplt = sns.displot(df.loc[:,['RandomForest','XGBoost','AdaBoost']], kind="kde")
    dplt.set(xlabel='Validation error', ylabel='Density', title='cross val scores distribution')
    plt.show()
    
    return



def main():  
    X_train, y_train, X_test, order_no_test = prepForModel('data/Train.csv', 
                                            'data/Test.csv', 'data/Riders.csv') 
    
    trainNet(torch.tensor(X_train), torch.tensor(y_train))
    trained_net = RegNet()
    trained_net.load_state_dict(torch.load('saved_model.pth'))
    predict(trained_net, X_test, 'predictions.csv', order_no_test)
    return


if __name__ == "__main__":
    main()

