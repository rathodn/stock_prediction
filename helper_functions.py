import json
from pandas.io.json import json_normalize

import pandas as pd
import numpy as np 
import requests

from datetime import date
from datetime import datetime, timedelta
from sklearn import metrics

from multiprocessing import Pool as ProcessPool 
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from numba import jit, njit, prange
from numba.typed import List as numbatypedList

from scipy.stats import linregress
import math
import time
from io import StringIO


import io

def encode_cyclical_feature(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    return df

def get_QM_ADR(df, window_size = 20 ,high_col = "Adj_High", low_col = "Adj_Low", close_col = "Adj_Close"):
    "function to calculate True Range and Average True Range"
    
    df['H_DIV_L'] = df[high_col]/df[low_col]
    
    df[f'ADR_percent_{str(window_size)}d'] = 100 * ( (df['H_DIV_L'].rolling(window_size).sum() - window_size) / window_size)
    
    return df.drop(['H_DIV_L'],axis=1).round(decimals=3)

def get_VWAP(df, groupbyCol, Vol_col, Price_col, window_size):
    
    df['prod'] = df[Vol_col] * df[Price_col]
    #df['VWAP_num'] = df.groupby(groupbyCol).apply(lambda x: x[["prod"]].rolling(window=window_size).sum() )
    df['VWAP_num'] = df["prod"].rolling(window_size).sum()
    #df['VWAP_denom'] = df.groupby(groupbyCol).apply(lambda x: x[[Vol_col]].rolling(window=window_size).sum() )
    df['VWAP_denom'] = df[Vol_col].rolling(window_size).sum()
    
    df['VWAP_' + Price_col + str(window_size)] = df['VWAP_num'] / df['VWAP_denom']
    df['VWAP_' + Price_col + str(window_size)] = np.where(df['VWAP_denom'] == 0,
                                                          0,
                                                          df['VWAP_' + Price_col + str(window_size)]
                                                         )
    df.drop(columns=['prod','VWAP_num', 'VWAP_denom'], inplace=True)
    
    return df.round(decimals=3)


def get_TR_ATR(df, window_size = 20 ,high_col = "Adj_High", low_col = "Adj_Low", close_col = "Adj_Close"):
    "function to calculate True Range and Average True Range"
    
    df['H-L'] = abs(df[high_col]-df[low_col])
    df['H-PC'] = abs(df[high_col]-df[close_col].shift(1))
    df['L-PC']= abs(df[low_col]-df[close_col].shift(1))
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR_' + str(window_size)] = df['TR'].rolling(window_size).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    
    return df.drop(['H-L','H-PC','L-PC'],axis=1).round(decimals=3)

def get_ADR(df, window_size = 20 ,high_col = "Adj_High", low_col = "Adj_Low", close_col = "Adj_Close"):
    "function to calculate True Range and Average True Range"
    
    df['H-L'] = abs(df[high_col]-df[low_col])
    
    df['DR_percent'] = 100 * (df['H-L'] / df[close_col])
    
    df[f'ADR_percent_{str(window_size)}_days'] = df['DR_percent'].rolling(window_size).mean()
    
    return df.drop(['H-L','DR_percent'],axis=1).round(decimals=3)

def get_BollBnd(df, window_size = 20, x_std_dev = 2, close_col = "Adj_Close" ):
    "function to calculate Bollinger Band" 
    
    df["MA"] = df[close_col].rolling(window_size).mean()
    df[f"BB_{x_std_dev}_up_{window_size}_{close_col}"] = df["MA"] + x_std_dev*df[close_col].rolling(window_size).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df[f"BB_{x_std_dev}_dn_{window_size}_{close_col}"] = df["MA"] - x_std_dev*df[close_col].rolling(window_size).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample 
    del df['MA']
    
    return df.round(decimals=3)

def get_RSI(df, window_size = 14, close_col = "Adj_Close"):
    "function to calculate RSI"
    delta = df[close_col].diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[window_size-1]] = np.mean( u[:window_size]) # first value is average of gains
    u = u.drop(u.index[:(window_size-1)])
    d[d.index[window_size-1]] = np.mean( d[:window_size]) # first value is average of losses
    d = d.drop(d.index[:(window_size-1)])
    rs = u.ewm(com=window_size,min_periods=window_size).mean()/d.ewm(com=window_size,min_periods=window_size).mean()
    df['RSI_' + str(window_size)] = (100 - 100 / (1+rs)).round(decimals=3)
    return df

def get_AGR(df, close_col = "Adj_Close", window_size = 252):
    " function to calculate Annual Growth Rate "
    df['beginning_value'] = df[close_col].shift(window_size)
    df['AGR_' + str(window_size)] = (df[close_col] / df['beginning_value']) - 1
    del df['beginning_value']
    df['AGR_' + str(window_size)] = df['AGR_' + str(window_size)].round(decimals=3)
    return df
    
def get_volatility(df, close_col = "Adj_Close", window_size = 252 ):
    " function to calculate volatility "
    df["daily_ret"] = df[close_col].pct_change()
    df['volatility_' + str(window_size)] = df["daily_ret"].rolling(window_size).std() * np.sqrt(window_size)
    del df['daily_ret']
    df['volatility_' + str(window_size)] = df['volatility_' + str(window_size)].round(decimals=3)
    return df 
    
def get_sharpe_ratio(df, rf = 0.022, volatility_col = "volatility_252", agr_col = "AGR_252", col_suffix='252'):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df['sharpe_ratio_' + col_suffix] = (df[agr_col] - rf) / df[volatility_col]
    df['sharpe_ratio_' + col_suffix] = df['sharpe_ratio_' + col_suffix].round(decimals=3)
    return df 

def get_max_drop(df, close_col = "Adj_Close", window_size = 252):
    
    @jit(parallel=True)
    def max_drop(vals):

        a = [0] + list(np.diff(vals) / vals[1:])
        a = np.asarray(a) + 1
        b = np.cumprod(a)
        b[:1] = 0
        c = np.maximum.accumulate(b)
        d = c-b 
        e = d/c 
        e[np.isnan(e)] = 0
        return max(e)
    
    "function to calculate max drawdown" 
    df["max_drop_"+ str(window_size)] = df[close_col].rolling(window_size).\
                                        apply(max_drop, engine='cython', raw=True )
    
    return df.round(3)

def get_calmar_ratio(df, agr_col = "AGR_252" , max_dd_col = "max_drop_252", window_size = 252):
    "function to calculate calmar ratio"
    df['calmar_' + str(window_size)] = df[agr_col] / df[max_dd_col]
    return df.round(3)

def convert_floats_to_float32(df):
    col_type = df.dtypes.reset_index()
    col_type.columns = ['Column_Name', 'type']
    float_columns = list(col_type["Column_Name"][(col_type['type']=="float64")
                                                ].values)
    for col in float_columns:
        #df[col] = np.round(df[col].astype(np.float32), N_decimals)
        df[col] = df[col].astype(np.float32)
        
    return df.round(decimals=3)


# get # and % of missing values in dataframe by columns
def missing_values_info(df):
    df_rows = df.shape[0]
    df = pd.DataFrame(df.isna().sum()).reset_index()        
    df.columns = ["Feature", "#missing_values"]
    df['%missing_values'] = df['#missing_values'] * 100 / df_rows
    df = df.sort_values(by=['#missing_values'],ascending=False).reset_index(drop=True)
    return df[df['#missing_values']>0]


# create columns/feature for NA representation
def create_feature_to_represent_NA(df, variable):
    # add additional variable to indicate missingness
    df[variable+'_NA'] = np.where(df[variable].isnull(), 1, 0)


# function to convert string type date column to date type
def str_to_date_col(col):
    df = col.drop_duplicates().reset_index(drop=True)
    col_name = col.columns[0]
    df.columns = [col_name + '_str']
    df['temp_date'] = df[col_name + '_str'].apply(lambda x: pd.to_datetime(x))
    col = pd.merge(col,df,how='left',left_on=[col_name],right_on=[col_name + '_str'])
    return col['temp_date']


def compute_error(yhat_train, y_train):
#     pdb.set_trace()
    indx = np.isnan(y_train) 
    print(sum(indx))
    out = metrics.mean_squared_error(yhat_train[~indx], y_train[~indx])
    out2 = metrics.mean_absolute_error(yhat_train[~indx], y_train[~indx])
    out3 = np.mean(np.abs((y_train[~indx] - yhat_train[~indx]) / (np.abs(y_train[~indx]) + 1) )) * 100

    print('rmse: ' + str(out**.5))
    print('mean_absolute_error: ' +  str(out2))
    print('mean_absolute_percentage_error: ' + str(out3))
    print('\n')
    

def run_function_in_multiprocessing(function_name, use_threads=False, pool_args = None, POOL_SIZE = 100): 

    if use_threads: 
        pool_cls = ThreadPool 
    else: 
        pool_cls = ProcessPool 

    with pool_cls(POOL_SIZE) as pool:
        results = pool.starmap(function_name, pool_args) 

    return results


def return_angle(values):
    slope = linregress(range(0,len(values)),  values ).slope 
    return math.degrees(math.atan( slope )) 

def change_column_order(df, col_name, index):
    cols = df.columns.tolist()
    cols.remove(col_name)
    cols.insert(index, col_name)
    return df[cols]
    
def requests_retry_session(
    # source = https://www.peterbe.com/plog/best-practice-with-retries-with-requests
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def get_data_from_API(tickers = [], func_to_run = None, func_args = None, return_dfs = 1, POOL_SIZE = 20,
                      WAIT_SECONDS = 61, chunk_size = 995, api_key = None ):

    df = pd.DataFrame()
    initial_time = time.time()
    
    res_list = []

    for i in range(0 , int(len(tickers)/chunk_size) + 1):
        loop_start_time = time.time()
        curr_chunk_start_time = time.time()

        start_list = i * chunk_size
        end_list = start_list + chunk_size

        if i < int(len(tickers)/chunk_size):
            partial_ticker_list = tickers[start_list:end_list]
        else:
            partial_ticker_list = tickers[start_list:len(tickers)]
        
        print("chunk ", i+1, ": ", len(partial_ticker_list))
        
        args = func_args
        
        results = run_function_in_multiprocessing(func_to_run, use_threads = True, 
                                                              pool_args = args, POOL_SIZE = POOL_SIZE)

        [res_list.append(result) for result in results]

        curr_chunk_end_time = time.time()

        if len(tickers) % chunk_size != len(partial_ticker_list): # need NOT to wait for the last loop
            while (curr_chunk_end_time - curr_chunk_start_time) < WAIT_SECONDS: # waiting for 1 min to complete before sending more requests
                time.sleep(1)
                curr_chunk_end_time = time.time()
        else:
            curr_chunk_end_time = time.time()
        #print(">>>>>total loop time: ", time.time() - loop_start_time)

    print("total time: ", time.time() - initial_time) 
    if return_dfs == 1:
        print("returning df")
        df = pd.concat(res_list).reset_index(drop=True)
    else:
        print("returning list of dfs")
        df = []
        for i in range(0,return_dfs):
            df.append(pd.concat([item[i] for item in res_list if item is not None]).reset_index(drop=True))
            
    return df 

def get_MA(df, groupbyCol, valueCol, window_size = 12):
    #return df.groupby(groupbyCol).apply(lambda x: x[[valueCol]].rolling(window=window_size).mean() ).round(decimals=3)
    return df[valueCol].rolling(window_size).mean().round(decimals=3)

def get_EMA(df, valueCol, window_size_list = [5,20]):
    #return df.groupby(groupbyCol).apply(lambda x: x[[valueCol]].rolling(window=window_size).mean() ).round(decimals=3)
    for i in window_size_list:
        df[f'EMA_{i}'] = df[valueCol].ewm(span=i).mean()
    return df

def get_std_dev(df, groupbyCol, valueCol, window_size = 12):
    #return df.groupby(groupbyCol).apply(lambda x: x[[valueCol]].rolling(window=window_size).std() ).round(decimals=3)
    return df[valueCol].rolling(window_size).std().round(decimals=3)

def get_MACD(df, LIST_fast_slow_MACD_Signal_cols, groupbyCol, valueCol, fast = 12, slow = 26, signal_span = 9):
    """ typical values fast = 12; slow =26, signal_span =9"""
    df[LIST_fast_slow_MACD_Signal_cols[0]] = get_MA(df, groupbyCol, valueCol, fast)
    df[LIST_fast_slow_MACD_Signal_cols[1]] = get_MA(df, groupbyCol, valueCol, slow)
    df[LIST_fast_slow_MACD_Signal_cols[2]] = df[LIST_fast_slow_MACD_Signal_cols[0]]-df[LIST_fast_slow_MACD_Signal_cols[1]] 
    #df[LIST_fast_slow_MACD_Signal_cols[3]] = df.groupby(groupbyCol).apply(lambda x: x[[LIST_fast_slow_MACD_Signal_cols[2]]].rolling(window=signal_span).mean() )
    df[LIST_fast_slow_MACD_Signal_cols[3]] = df[LIST_fast_slow_MACD_Signal_cols[2]].rolling(window=signal_span).mean()
    
    return df.round(decimals=3)

def get_OBV(DF, groupbyCol, valueCol):
    
    #df = DF.copy()
    df = DF[[groupbyCol,valueCol]]
    
    df['percentage_change'] =  df.groupby(groupbyCol)[valueCol].pct_change()
    df['direction'] = np.where(df['percentage_change']>=0,1,-1)
    df['direction'] = np.where(pd.isnull(df['percentage_change']),0,df['direction'])
    df['value_direction'] = df[valueCol] * df['direction']
    
    return df.groupby(groupbyCol)['value_direction'].cumsum().round(decimals=3)

def create_set_of_standard_features(df, groupbyCol, base_feature):
    
    df[ base_feature + '_lag1'] = df.groupby(groupbyCol)[base_feature].shift(1)
    
    df = get_MACD(df, [base_feature + "_MA12", base_feature + "_MA26", base_feature + "_MACD_12_26", 
                              base_feature + "_MACD_12_26_Signal_9" ], groupbyCol, base_feature, 12, 26, 9)

    df = get_MACD(df, [base_feature + "_MA3", base_feature + "_MA5", base_feature + "_MACD_3_5", 
                                base_feature + "_MACD_3_5_Signal_2" ], groupbyCol, base_feature, 3, 5, 2)

    df = get_MACD(df, [base_feature + "_MA40", base_feature + "_MA90", base_feature + "_MACD_40_90", 
                                base_feature + "_MACD_40_90_Signal_35" ], groupbyCol, base_feature, 40, 90, 35)

    for i in [200, 150, 100, 50, 20, 15, 10, 2]:
        df[base_feature + '_MA' + str(i)] =  get_MA(df, groupbyCol, base_feature, i)

    for i in [150, 50, 15, 5, 2]:
        df[base_feature + '_STD' + str(i)] =  get_std_dev(df, groupbyCol, base_feature, i)
    
    return df.round(decimals=3) 

def create_set_of_standard_features_ana(df, groupbyCol, base_feature):
    
    df[ base_feature + '_lag1'] = df.groupby(groupbyCol)[base_feature].shift(1)
    
    for i in [200, 150, 100, 90,50, 40,26,20, 15, 12,10, 5,3,2]:
        df[base_feature + '_MA' + str(i)] =  get_MA(df, groupbyCol, base_feature, i)
    
    return df.round(decimals=3) 

def create_set_of_standard_Volume_features(df, groupbyCol, base_feature):
    
    df[ base_feature + '_lag1'] = df.groupby(groupbyCol)[base_feature].shift(1)
    
    for i in [60, 20, 10, 5, 2]:
        df[base_feature + '_MA' + str(i)] =  get_MA(df, groupbyCol, base_feature, i)

    for i in [20, 10, 5, 2]:
        df[base_feature + '_STD' + str(i)] =  get_std_dev(df, groupbyCol, base_feature, i)
    
    df['On_Balance_' + base_feature] = get_OBV(df, groupbyCol , base_feature)
    
    return df.round(decimals=3) 


def get_N_days_stat(df, window_days, value_col, groupby_cols):
    
    df['shift'] = df.groupby(groupby_cols)[value_col].shift(1)
    
    df[value_col + '_low_' + str(window_days) + '_days'] = df["shift"].rolling(window_days).min()
    df[value_col + '_high_' + str(window_days) + '_days'] = df["shift"].rolling(window_days).max()
    
    df = get_diff_per_feature_single(df, value_col, value_col + '_low_' + str(window_days) + '_days')
    df = get_diff_per_feature_single(df, value_col, value_col + '_high_' + str(window_days) + '_days')
    
    df.drop(columns=['shift',value_col + '_low_' + str(window_days) + '_days', value_col + '_high_' + str(window_days) + '_days'], inplace=True)
    
    return df.round(decimals=3)

def create_lag_feature_for_DFuture(df, groupbyCol, col_name, days_in_future = 1):
    for lagN in range(1, days_in_future):
        df[f'{col_name}_DFuture{lagN}'] = df.groupby(groupbyCol)[col_name].shift(lagN) 
    
    return df.groupby(['Ticker']).apply(lambda x: x.iloc[days_in_future-1:]) 
