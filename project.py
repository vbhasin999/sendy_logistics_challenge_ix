import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


def clean_data(df: pd.DataFrame, cols_to_drop: list):
    """takes in a pandas.DataFrame object and a list of columns to remove 
    removes the respective columns inplace 

    Arguments:
        df (pandas.DataFrame): dataframe from which columns will be removed
        cols_to_drop (list): list of column names to be dropped
    
    Returns:
        None
    """
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
                        'Arrival at Destination - Time']
    test_cols_to_drop = ['Order No', 'User Id', 'Vehicle Type']
    riders_cols_to_drop = []

    # removing columns from the dataframes
    clean_data(train, train_cols_to_drop)
    clean_data(test, test_cols_to_drop)
    clean_data(riders, riders_cols_to_drop)

    # merging train and test dataframes with rider data
    m_train = merge_dataframes(train, riders)
    m_test = merge_dataframes(test, riders)



    print(f"PROCESSED DATA: \ntrain:\n{m_train.head()}\ntest:\n{m_test.head()}")
    return


if __name__ == "__main__":
    main()


