import pandas as pd
import numpy as np
import pytz
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pytz import timezone
import matplotlib.lines as mlines
from scipy import stats
import plotly.offline as py
import plotly.graph_objs as go
from util.date.date_aux import convert_date
from util.data.data_aux import get_feature,join_df
from Scripts.PHQ9 import getPhq9

user_id = ['u00','u01','u02','u04','u05','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22',
        'u23','u25','u27','u30','u31','u32','u33','u34','u35','u36','u41','u42','u43','u44','u45'
    ,'u46','u51','u52','u54','u57','u58','u59']
#user_id = ['u03']

#verificar a quantidade de dados da última semana

coll_agreg = ['all_dur_day','week01_dur_day','week02_dur_day','week03_dur_day','week04_dur_day',
              'week05_dur_day', 'week06_dur_day', 'week07_dur_day', 'week08_dur_day']

coll_week = ['ALL', 'Week01', 'Week02', 'Week03', 'Week04','Week05','Week06','Week07','Week08']

coll_feature=['mean', 'median', 'max', 'std', 'var', 'mad']

coll_agreg_feature = pd.MultiIndex.from_product([coll_agreg, coll_feature])


def read_data(name):
    lock_data = pd.read_csv('../SensingData/PhoneLock/phonelock_'+name+'.csv')
    return lock_data

df_all = pd.DataFrame(data=None, index=user_id, columns=coll_agreg_feature)

for uid in user_id:
    lock_data = read_data(uid)
    convert_date(lock_data, columns=['start', 'end'])
    lock_data.index = lock_data['start']
    lock_data['duration'] = pd.to_timedelta(lock_data['end'] - lock_data['start']).dt.seconds/3600
    #conv_data['week_day'] = conv_data['start_timestamp'].apply(lambda x: x.weekday())
    lock_data['week_day'] = lock_data.index.dayofweek
    lock_data['week_group'] = lock_data['start'].dt.to_period('W-THU')

    all = lock_data.resample('D')
    #print(all[all['duration']<2]['duration'])

    week01 = pd.concat(
        join_df('2013-03-22/2013-03-28', '2013-03-29/2013-04-04', '2013-04-05/2013-04-11', lock_data)).resample('D')
    week02 = pd.concat(join_df('2013-04-05/2013-04-11', '2013-04-12/2013-04-18', None, lock_data)).resample('D')
    week03 = pd.concat(join_df('2013-04-12/2013-04-18', '2013-04-19/2013-04-25', None, lock_data)).resample('D')
    week04 = pd.concat(join_df('2013-04-19/2013-04-25', '2013-04-26/2013-05-02', None, lock_data)).resample('D')
    week05 = pd.concat(join_df('2013-04-26/2013-05-02', '2013-05-03/2013-05-09', None, lock_data)).resample('D')
    week06 = pd.concat(join_df('2013-05-03/2013-05-09', '2013-05-10/2013-05-16', None, lock_data)).resample('D')
    week07 = pd.concat(join_df('2013-05-10/2013-05-16', '2013-05-17/2013-05-23', None, lock_data)).resample('D')
    week08 = pd.concat(join_df('2013-05-17/2013-05-23', '2013-05-24/2013-05-30', None, lock_data)).resample('D')

    all_lock_dur = get_feature((all['duration'].sum()))

    week01_lock_dur = get_feature((week01['duration'].sum()))

    week02_lock_dur = get_feature((week02['duration'].sum()))

    week03_lock_dur = get_feature((week03['duration'].sum()))

    week04_lock_dur = get_feature((week04['duration'].sum()))

    week05_lock_dur = get_feature((week05['duration'].sum()))

    week06_lock_dur = get_feature((week06['duration'].sum()))

    week07_lock_dur = get_feature((week07['duration'].sum()))

    week08_lock_dur = get_feature((week08['duration'].sum()))

    list_feature = all_lock_dur + week01_lock_dur + week02_lock_dur + week03_lock_dur + \
                   week04_lock_dur + week05_lock_dur + week06_lock_dur + \
                   week07_lock_dur + week08_lock_dur
    df_all.index.name = 'uid'
    df_all.loc[uid] = list_feature

#carrega o df do phq9
df_phq9=getPhq9()
df_lock_phq9 = pd.merge(df_all, df_phq9, how='inner',on='uid')

#df contendo as correlações
df_crr = pd.DataFrame(data=None, index=coll_agreg_feature, columns=['pearson_base', 'p_value_base', 'pearson_follow', 'p_value_follow'])


def creat_corr(df, index):
    pearson_base=[]
    pearson_follow = []
    p_base=[]
    p_follow=[]
    for x in coll_feature:
        pearson_coef, p_value = stats.pearsonr(df_lock_phq9['phq9','final_score_x'], df[x])
        pearson_base.append(pearson_coef)
        p_base.append(p_value)
        pearson_coef, p_value = stats.pearsonr(df_lock_phq9['phq9', 'final_score_y'], df[x])
        pearson_follow.append(pearson_coef)
        p_follow.append(p_value)
    df_crr.loc[index, 'pearson_base']= np.array(pearson_base)
    df_crr.loc[index, 'pearson_follow'] = np.array(pearson_follow)
    df_crr.loc[index, 'p_value_base'] = np.array(p_base)
    df_crr.loc[index, 'p_value_follow'] = np.array(p_follow)
#print(df_bt_phq9)

for x in coll_agreg:
    creat_corr(df_lock_phq9[x],x)
    #print(df_crr)

df_crr = df_crr.T
df_crr = df_crr.loc[:, pd.IndexSlice[:, ['mean', 'median', 'max', 'std', 'var', 'mad']]].T.reset_index()


def create_chart(statist='mean', pearson='pearson_base', name=None, color = '#341f97'):
    df_crr_var = df_crr[df_crr['level_1'] == statist]
    df_crr_var['semana'] = coll_week
    # Gráfico usando apenas marcadores
    trace = go.Scatter(x=df_crr_var['semana'],
                        y=df_crr_var[pearson],
                        mode='markers+lines',
                        name=name,
                        line={'color': color},
                        )
    return trace

trace1 = create_chart('median','pearson_follow', 'Mediana', '#ee5253' )
trace2 = create_chart('mean','pearson_follow', 'Média', '#009900')
trace3 = create_chart('var','pearson_follow', 'Variância', '#341f97')
trace4 = create_chart('max','pearson_follow', 'Máximo', '#8B4513')
trace5 = create_chart('std','pearson_follow', 'STD', '#8B008B')
trace6 = create_chart('mad','pearson_follow', 'Mad', '#800000')
data = [trace1, trace2,trace3, trace4, trace5, trace6]

layout = go.Layout(title = 'Correlações Temporais - Duração de Atividades',
                   titlefont = {'family': 'Arial',
                                'size': 22,
                                'color': '#7f7f7f'},
                   xaxis = {'title': 'Período'},
                    yaxis=dict(
                            title='Coeficiente de Correlação (r)',
                            autorange=False,
                            showgrid=False,
                            zeroline=False,
                            range=[0.2, -0.7]

                        ),
                   paper_bgcolor = 'rgb(243, 243, 243)',
                   plot_bgcolor = 'rgb(243, 243, 243)')
fig = go.Figure(data=data, layout=layout)


py.plot(fig)











