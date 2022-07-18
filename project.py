import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataProcessing import *
#rename column names so that they don't have spaces
#remove duplicates 
#modify 'time' values, pd.to_datetime? research pandas methods to work w time
#missing values 

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
                        'Confirmation - Time']

    test_cols_to_drop = ['Order No', 'User Id', 'Vehicle Type',
                        'Placement - Day of Month',
                        'Placement - Weekday (Mo = 1)',
                        'Placement - Time',
                        'Confirmation - Day of Month', 
                        'Confirmation - Weekday (Mo = 1)',
                        'Confirmation - Time',
                        'Arrival at Pickup - Time']
    riders_cols_to_drop = []

    # time columns remaining which need to be converted to datetime objects
    time_cols = ['Pickup - Time']

    # columns which contain categorical variables
    categorical_cols = []

    # removing columns from the dataframes
    clean_data(train, train_cols_to_drop, time_cols)
    clean_data(test, test_cols_to_drop, time_cols)
    clean_data(riders, riders_cols_to_drop, [])

    # merging train and test dataframes with rider data
    m_train = merge_dataframes(train, riders)
    m_test = merge_dataframes(test, riders)

 
    print(f"m_train: \n{m_train.columns}")
    '''
    sns.relplot(data=m_train, x='Age', y='Time from Pickup to Arrival', hue='Rider Id')
    sns.relplot(data=m_train, x='No_Of_Orders', y='Time from Pickup to Arrival', hue='Rider Id')
    sns.relplot(data=m_train, x='No_of_Ratings', y='Time from Pickup to Arrival', hue='Rider Id')
    '''
    sns.heatmap(data=m_train.loc[:,['Platform Type', 'Arrival at Pickup - Day of Month',
       'Arrival at Pickup - Weekday (Mo = 1)',
       'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)', 
       'Distance (KM)', 'Temperature', 'Precipitation in millimeters',
       'Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long',
       'No_Of_Orders', 'Age','Average_Rating', 'No_of_Ratings',
       'Time from Pickup to Arrival']].corr(), square=True)
    plt.show()
    return


if __name__ == "__main__":
    main()


