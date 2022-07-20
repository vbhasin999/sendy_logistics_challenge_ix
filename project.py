import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataProcessing import *

#missing values 
#feature engineering: rider tiers? keep original features after making new column?
#low corr for rider features

def main():
    train_file = "data/Train.csv"
    test_file = "data/Test.csv"
    rider_file = "data/Riders.csv"
    train, test, riders = load_data(train_file, test_file, rider_file)


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
                        'Pickup - Day of Month',
                        'Precipitation in millimeters'
                        ]

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
                        'Pickup - Day of Month',
                        'Precipitation in millimeters']

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

    # feature engineering
    m_train = FENG_TODcol(m_train)
    m_train = FENG_weekend(m_train)

    print(f"FINAL SELECTED FEATURES: \n{m_train.columns}")

    # columns which contain categorical variables
    categorical_cols = ['Personal or Business','Platform Type', 
     'Pickup - Weekday (Mo = 1)', 'Pickup - Time', 'TOD', 'weekend']

    sns.heatmap(m_train[['No_Of_Orders', 'Age', 'Average_Rating',
       'No_of_Ratings', 'Time from Pickup to Arrival']].corr(), square=True)\

    
    sns.catplot(data=m_train,x='TOD',y='Time from Pickup to Arrival',col='weekend')
    sns.catplot(data=m_train,x='Pickup - Weekday (Mo = 1)',
    y='Time from Pickup to Arrival', col='TOD')
    sns.catplot(data=m_train,x='weekend',y='Time from Pickup to Arrival')
    sns.catplot(data=m_train,x='TOD',y='Time from Pickup to Arrival')
    plt.show()

    m_train = oneHotEncode(m_train, categorical_cols)
    print(f"FINAL FEATURES AFTER ONE HOT ENCODING: \n{m_train.columns}")    
    return


if __name__ == "__main__":
    main()

