import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler


###############################################################################
# file which contains functions to process the data and modify features
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
    new_col = imp.fit_transform(df[colName].to_frame())
    df[colName] = new_col
    return

def remove_time_outliers(df: pd.DataFrame):
    filtered_df = df.loc[(df['Pickup - Time'] >= pd.to_datetime('06:00:00'))& \
        (df['Pickup - Time'] <= pd.to_datetime('23:00:00'))]
    
    return filtered_df

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
        (train_X, train_y, test_X) as np.ndarray types
    """ 
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

    # getting the order numbers for the test dataset
    order_no_test = test['Order No'].to_frame()

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

    # only keep rows where the pickup time is between 6AM and 11PM (inclusive)
    m_train = remove_time_outliers(m_train)

    # feature engineering
    m_train = FENG_TODcol(m_train)
    m_train = FENG_weekend(m_train)

    m_test = FENG_TODcol(m_test)
    m_test = FENG_weekend(m_test)
    
    imputeMissingVals(m_train, 'Temperature')
    imputeMissingVals(m_test, 'Temperature')

    # columns which contain categorical variables
    categorical_cols = ['Personal or Business','Platform Type', 
     'Pickup - Weekday (Mo = 1)', 'Pickup - Time', 'TOD', 'weekend']

    
    m_train = oneHotEncode(m_train, categorical_cols)
    m_test = oneHotEncode(m_test, categorical_cols)
    
    # separating labels and features
    y = m_train['Time from Pickup to Arrival'].ravel() #.to_frame()
    X = m_train.drop('Time from Pickup to Arrival', axis=1)
   
    scaler = StandardScaler()
    # Fit the scaler object to the training data and then standardise.
    X_train = scaler.fit_transform(X)
    
    X_train = X_train.astype(np.float32)

    # Standardise the testing data using the same scaler object.
    X_test = scaler.transform(m_test)
    X_test = X_test.astype(np.float32)

    y = y.astype(np.float32)

    return (X_train, y, X_test, order_no_test)
