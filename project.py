import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#rename column names so that they don't have spaces
#remove duplicates 
#modify 'time' values, pd.to_datetime? research pandas methods to work w time
#missing values 
def load_data(train_file: str, test_file: str, rider_file: str):
    """takes in pathnames to the train, test and rider data files and 
    returns pandas.DataFrame objects for the three datasets respectively
    
    Arguments:
        train_file (str): pathname of csv file with train data
        test_file (str): pathname of csv file with test data
        rider_file (str): pathname of csv file with rider data
    
    Returns:
        (train, test, riders) {tuple}
        train {pandas.DataFrame}: train dataframe
        test {pandas.DataFrame}: test dataframe
        riders {pandas.DataFrame}: rider dataframe
        
    """
    
    
    train = pd.read_csv('data/Train.csv')
    test = pd.read_csv('data/Test.csv')
    riders = pd.read_csv('data/Riders.csv')
    

    return (train, test, riders)


def clean_data(df: pd.DataFrame, cols_to_drop: list, time_cols: list):
    """takes in a pandas.DataFrame object, list of columns to remove and a list
    of column names with time data. Does the following:
        1.removes duplicate rows in the data
        2.converts the time columns of the data to pandas.datetime64 objects  
        3.rounds the time to the nearest hour 
        4.removes the respective columns inplace
    
    Arguments:
        df (pandas.DataFrame): dataframe from which columns will be removed
        cols_to_drop (list): list of column names to be dropped
        time_cols (list): list of column names with time data to be converted
    
    Returns:
        None
    """
    df.drop_duplicates(inplace=True)
    
    for time_col in time_cols:
        df[time_col] = (pd.to_datetime(df[time_col]))
        df[time_col] = df[time_col].dt.round('1H')

    df.drop(columns=cols_to_drop, inplace=True)
    return

def merge_dataframes(standard_data: pd.DataFrame, rider_data: pd.DataFrame):
    """takes in a standard (train/test) dataframe and the rider dataframe and
    returns the merged dataframe

    Args:
        standard_data (pandas.DataFrame): train/test dataframe
        rider_data (pandas.DataFrame): rider dataframe

    Returns:
        pandas.DataFrame: merged dataframe where each row contains the standard
        features and the corresponding rider features
    """
    return pd.merge(standard_data,rider_data,on="Rider Id")

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

    # removing columns from the dataframes
    clean_data(train, train_cols_to_drop, time_cols)
    clean_data(test, test_cols_to_drop, time_cols)
    clean_data(riders, riders_cols_to_drop, [])

    # merging train and test dataframes with rider data
    m_train = merge_dataframes(train, riders)
    m_test = merge_dataframes(test, riders)

 
    print(f"m_train: \n{m_train.columns}")

    sns.relplot(data=m_train, x='Age', y='Time from Pickup to Arrival', hue='Rider Id')
    sns.relplot(data=m_train, x='No_Of_Orders', y='Time from Pickup to Arrival', hue='Rider Id')
    sns.relplot(data=m_train, x='No_of_Ratings', y='Time from Pickup to Arrival', hue='Rider Id')
    plt.show()
    return


if __name__ == "__main__":
    main()


