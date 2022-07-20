import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsClassifier as KNN

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
    return df

def imputeMissingVals(df: pd.DataFrame, colName: str):
    """imputes the missing values for a single column and drops the original

    Args:
        df (pd.DataFrame): dataFrame
        colName (str): name of column to be dropped
    
    Returns:
        None
    """    
    imp = IterativeImputer(random_state=0)
    new_temp = imp.fit_transform(df[colName].to_frame())
    df[colName] = new_temp
    return
def FENG_weekend(df: pd.DataFrame):
    """creates a column specifiying if pickup was on a weekend or not

    Args:
        df (pd.DataFrame): dataframe
    
    Returns:
        df
    """     
    df['weekend'] = df['Pickup - Weekday (Mo = 1)'] >= 6
    return df

def FENG_TODcol(df: pd.DataFrame):
    """adds a column which specifies the time of day of pick up from one of four
    categories: morning, afternoon, evening, night

    Args:
        df (pd.DataFrame): dataframe

    Returns:
        df

    """   

    #df.loc[(df['Pickup - Time'] >= pd.to_datetime('5:00:00')) & (df['Pickup - Time'] <= pd.to_datetime('5:00:00')), :]["TOD"] = "morning"
    
    conditions = [
   (df['Pickup - Time'] >= pd.to_datetime('05:00:00')) & (df['Pickup - Time'] <= pd.to_datetime('11:59:00')),
   (df['Pickup - Time'] >= pd.to_datetime('12:00:00')) & (df['Pickup - Time'] <= pd.to_datetime('17:59:00')),
   (df['Pickup - Time'] >= pd.to_datetime('18:00:00')) & (df['Pickup - Time'] <= pd.to_datetime('19:59:00')),
   (df['Pickup - Time'] >= pd.to_datetime('20:00:00')) | (df['Pickup - Time'] <= pd.to_datetime('4:59:00')),
   ]

    values = ['morning','afternoon','evening','night']

    df['TOD'] = np.select(conditions, values)
    return df

def prepForModel(train_file: str, test_file: str, rider_file: str):
    """Wrapper function which performs all the necessary data processing
    to prepare the data for the ML model. cleans data, merges dataframes,
    imputes missing data, adds engineered feature columns and one hot encodes
    categorical variables

    Args:
        train_file (str): pathname to train data file
        test_file (str): pathname to test data file
        rider_file (str): pathname to rider data file
    
    Returns:
        (train_X, train_y, test_X)
    """    
    return