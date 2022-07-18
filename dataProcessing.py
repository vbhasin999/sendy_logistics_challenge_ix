import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
###############################################################################
# file which contains functions to process teh data and modify features
###############################################################################
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

def oneHotEncode(df: pd.DataFrame, colNames: list):
    print(f"before: \n{df.columns}")
    """takes in a pandas.DataFrame and a list of column names of categorical 
    variables, converts the columns to one hot encoded columns

    function taken from:
    https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33

    Args:
        df (pandas.DataFrame): dataframe 
        colNames (list): list of column names to be converted to one hot

    Returns:
        pandas.DataFrame: modified dataframe
    """    
    for col in colNames:
        dummies = pd.get_dummies(df[col],prefix=col)
        df = pd.concat([df,dummies],axis=1)

        #drop the encoded column
        df.drop([col],axis = 1 , inplace=True)
        
    print(f"after: \n{df.head()}")
    return df
