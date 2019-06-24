import pandas as pd
import numpy as np
import pytz
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pytz import timezone
import os
import re
import pprint
import inspect
conversation_path = '../SensingData/Conversations/'

user_id = ['u00','u01','u02','u03','u04','u05','u07','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22',
           'u23','u24','u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44','u45','u46','u47','u49',
           'u50','u51','u52','u53','u54','u56','u57','u58','u59']

def convert_date(df, columns=['timestamp'], time_range=False):
    """ # input:
        df: original dataframe.
        columns: represent names of columns that was time type.
        change_index: True for audio and activity data, False for conversation data
        # output: dataframe. Turn timestamp from unix time into readable time. and set index and change time zone.
    """
    for col in columns:
        df[col] = pd.to_datetime(df[col], unit='s', utc=True).dt.tz_convert(
            'US/Eastern')  # .dt.tz_localize('US/Eastern')
        if not time_range:
            # there is only on timestamp column, so use it as index
            df.index = pd.to_datetime(df[col])  # .dt.date
    return df

def custom_resampler(array_like):
    if len(array_like)>0:
        np.sum(array_like)/len(array_like)
        return np.sum(array_like)/len(array_like)
    else:
        return pd.to_timedelta('0 days')

# test time_range_create_df
u01_conv = pd.read_csv(conversation_path+'conversation_u01.csv')
u01_conv.columns = ['start_timestamp','end_timestamp']
df_user = convert_date(u01_conv,['start_timestamp','end_timestamp'],True)
df_user.index = df_user['start_timestamp']

df_user['duration'] = pd.to_timedelta(df_user['end_timestamp'] - df_user['start_timestamp'])
# print('ei',df_user['duration'].mean(),df_user.index)

# get total times of coversation
cov_freq = df_user['duration'].resample('D').apply(custom_resampler)
df_user[(df_user.index.day==2) & (df_user.index.month==5)]['duration'],cov_freq


def custom_resampler(array_like):
    if len(array_like) > 0:

        return np.sum(array_like) / len(array_like)
    else:
        return pd.to_timedelta('0 days')

def read_data(name):
    conversation_data = pd.read_csv('../SensingData/Conversations/conversation_'+name+'.csv')
    return conversation_data

def time_range_create_df(path, unit='H', time_range=False, agg_method='mean'):
    """
    # input:
    path: path of directory of all files.
    unit: whether H for hourly or D for Daily
    change_index: True for audio and activity data, False for conversation data
    """
    f_names = user_id
    new_time_df = pd.DataFrame()

    for user_f in f_names:

        data_user = read_data(path)

        # new for range data
        if time_range:

            data_user.columns = ['start_timestamp', 'end_timestamp']
            df_user = convert_date(data_user, ['start_timestamp', 'end_timestamp'], time_range)
            df_user.index = df_user['start_timestamp']

            if str(agg_method) == 'mean':
                df_user['duration'] = pd.to_timedelta(df_user['end_timestamp'] - df_user['start_timestamp'])

                # get total times of coversation
                cov_freq = df_user['duration'].resample(unit).apply(custom_resampler)

                # turn a series of data into a row(with dataframe type).
                cov_freq_transposed = cov_freq.to_frame(
                    name=re.sub(r'^conversation_(.+)\.csv$', '\g<1>', user_f)).transpose()

            elif str(agg_method) == 'max':
                df_user['duration'] = pd.to_timedelta(df_user['end_timestamp'] - df_user['start_timestamp'])

                # get total times of coversation
                cov_freq = df_user['duration'].resample(unit).max()

                # turn a series of data into a row(with dataframe type).
                cov_freq_transposed = cov_freq.to_frame(
                    name=re.sub(r'^conversation_(.+)\.csv$', '\g<1>', user_f)).transpose()

            elif str(agg_method) == 'count':
                # get total times of coversation
                cov_freq = df_user['start_timestamp'].resample(unit).count()

                # turn a series of data into a row(with dataframe type).
                cov_freq_transposed = cov_freq.to_frame(
                    name=re.sub(r'^conversation_(.+)\.csv$', '\g<1>', user_f)).transpose()

            # add into final result
            new_time_df = pd.concat([new_time_df, cov_freq_transposed])
        else:

            df_user = convert_date(data_user)

            # select '*** inference' column name
            col_inf = [i for i in df_user.columns if 'inference' in i][0]

            if str(agg_method)== 'drop':
                # remove_unrecognized_category: '3: Unknown'
                df_user = df_user[(df_user[col_inf] != 3)]

            # group second data into days.
            weekly = df_user[col_inf].resample(unit).mean()
            # weekly.plot(style = [':','--','-'])

            # turn a series of data into a row(with dataframe type).
            weekly_transposed = weekly.to_frame(name=user_f.strip('.csv')).transpose()

            # add into final result
            new_time_df = pd.concat([new_time_df, weekly_transposed])
    return new_time_df


conversation_time_df_day_count = time_range_create_df('u00', unit='D', time_range=True, agg_method='count')

conversation_time_df_day_mean = time_range_create_df('u00', unit='D', time_range=True, agg_method='mean')

re.sub(r'^conversation_(.+)\.csv$', '\g<1>', 'conversation_u02.csv')

sns.scatterplot(data=conversation_time_df_day_count, x='')

print(conversation_time_df_day_count.transpose().head(100))