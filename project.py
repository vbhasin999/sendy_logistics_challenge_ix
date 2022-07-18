import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataProcessing import *
#rename column names so that they don't have spaces
#missing values 
#feature engineering: find a better way to use day of month, maybe merge
#average rating and number of ratings?

def main():
    train_file = "data/Train.csv"
    test_file = "data/Test.csv"
    rider_file = "data/Riders.csv"
    train, test, riders = load_data(train_file, test_file, rider_file)

    # feature engineering: making a column which specifies if the pickup
    # was on a weekend
    train['weekend'] = train['Pickup - Weekday (Mo = 1)'] >= 6
    test['weekend'] = test['Pickup - Weekday (Mo = 1)'] >= 6

    # specifying which columns to drop from the standard and rider dataframes
    train_cols_to_drop = ['Order No', 'User Id', 'Vehicle Type', 
                        'Arrival at Destination - Day of Month',
                        'Arrival at Destination - Weekday (Mo = 1)',
                        'Arrival at Destination - Time',
                        'Placement - Day of Month',
                        'Placement - Weekday (Mo = 1)',
                        'Placement - Time',
                        'Confirmation - Day of Month', 
                        'Confirmation - Weekday (Mo = 1)',
                        'Confirmation - Time',
                        'Arrival at Pickup - Day of Month', 
                        'Arrival at Pickup - Weekday (Mo = 1)',
                        'Arrival at Pickup - Time',
                        'Pickup - Day of Month',]

    test_cols_to_drop = ['Order No', 'User Id', 'Vehicle Type',
                        'Placement - Day of Month',
                        'Placement - Weekday (Mo = 1)',
                        'Placement - Time',
                        'Confirmation - Day of Month', 
                        'Confirmation - Weekday (Mo = 1)',
                        'Confirmation - Time',
                        'Arrival at Pickup - Day of Month', 
                        'Arrival at Pickup - Weekday (Mo = 1)',
                        'Arrival at Pickup - Time',
                        'Pickup - Day of Month',]

    riders_cols_to_drop = []

    # time columns remaining which need to be converted to datetime objects
    time_cols = ['Pickup - Time']

    # removing columns from the dataframes
    clean_data(train, train_cols_to_drop, time_cols)
    clean_data(test, test_cols_to_drop, time_cols)
    clean_data(riders, riders_cols_to_drop, [])

    # merging train and test dataframes with rider data
    m_train = merge_dataframes(train, riders)
    m_test = merge_dataframes(test, riders)

    # dropping rider ID column after dataframes have been merged
    m_train.drop(columns=['Rider Id'], inplace=True)
    m_test.drop(columns=['Rider Id'], inplace=True)

    print(f"FINAL FEATURES: \n{m_train.columns}")

    # columns which contain categorical variables
    categorical_cols = ['Personal or Business','Platform Type', 
     'Pickup - Weekday (Mo = 1)', 'Pickup - Time']
    
    m_train = oneHotEncode(m_train, categorical_cols)
    return


if __name__ == "__main__":
    main()

