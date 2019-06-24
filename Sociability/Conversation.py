import pandas as pd
import numpy as np
import pytz
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sc
from scipy import stats
from Scripts.PHQ9 import getPhq9
import plotly.offline as py
import plotly.graph_objs as go
from util.date.date_aux import convert_date
from util.data.data_aux import get_feature,join_df

#remover frequencia zerada

user_id = ['u00','u01','u02','u04','u05','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22',
        'u23','u25','u27','u30','u31','u32','u33','u34','u35','u36','u41','u42','u43','u44','u45'
    ,'u46','u51','u52','u54','u57','u58','u59']

#tratar zeros e nan -> ,'u03','u39'

coll_agreg = ['all_freq_day', 'all_dur_day', 'week01_freq_day','week01_dur_day','week02_freq_day','week02_dur_day',
                'week03_freq_day', 'week03_dur_day','week04_freq_day','week04_dur_day','week05_freq_day','week05_dur_day',
                'week06_freq_day', 'week06_dur_day','week07_freq_day','week07_dur_day','week08_freq_day','week08_dur_day']

coll_feature=['mean', 'median', 'max', 'std', 'var', 'mad']

coll_agreg_feature = pd.MultiIndex.from_product([coll_agreg, coll_feature])

coll_week = ['ALL', 'Week01', 'Week02', 'Week03', 'Week04','Week05','Week06','Week07','Week08']


def read_data(name):
    conversation_data = pd.read_csv('../SensingData/Conversations/conversation_'+name+'.csv')
    return conversation_data


df_all = pd.DataFrame(data=None, index=user_id, columns=coll_agreg_feature)


for uid in user_id:
    conv_data = read_data(uid)
    convert_date(conv_data, columns=['start_timestamp', 'end_timestamp'])
    conv_data.index = conv_data['start_timestamp']
    conv_data['duration'] = pd.to_timedelta(conv_data['end_timestamp'] - conv_data['start_timestamp']).dt.seconds
    #conv_data['week_day'] = conv_data['start_timestamp'].apply(lambda x: x.weekday())
    conv_data['week_day'] = conv_data.index.dayofweek
    conv_data['week_group'] = conv_data['start_timestamp'].dt.to_period('W-THU')

    all = conv_data.resample('D')

    week01 = pd.concat(join_df('2013-03-22/2013-03-28','2013-03-29/2013-04-04','2013-04-05/2013-04-11',conv_data)).resample('D')
    week02 = pd.concat(join_df('2013-04-05/2013-04-11', '2013-04-12/2013-04-18',None,conv_data)).resample('D')
    week03 = pd.concat(join_df('2013-04-12/2013-04-18','2013-04-19/2013-04-25',None,conv_data)).resample('D')
    week04 = pd.concat(join_df('2013-04-19/2013-04-25','2013-04-26/2013-05-02',None,conv_data)).resample('D')
    week05 = pd.concat(join_df('2013-04-26/2013-05-02', '2013-05-03/2013-05-09',None,conv_data)).resample('D')
    week06 = pd.concat(join_df('2013-05-03/2013-05-09','2013-05-10/2013-05-16',None,conv_data)).resample('D')
    week07 = pd.concat(join_df('2013-05-10/2013-05-16','2013-05-17/2013-05-23',None,conv_data)).resample('D')
    week08 = pd.concat(join_df('2013-05-17/2013-05-23','2013-05-24/2013-05-30',None,conv_data)).resample('D')
    print(week05.count())

    #aux2 = conv_data[conv_data['week_group'] == '2013-05-31/2013-06-06']

    all_conv_freq = get_feature(all['duration'],'count')
    all_conv_dur = get_feature(all['duration'], 'sum')

    week01_conv_freq = get_feature(week01['duration'], 'count')
    week01_conv_dur = get_feature(week01['duration'], 'sum')

    week02_conv_freq = get_feature(week02['duration'], 'count')
    week02_conv_dur = get_feature(week02['duration'],'sum')

    week03_conv_freq = get_feature(week03['duration'],'count')
    week03_conv_dur = get_feature(week03['duration'],'sum')

    week04_conv_freq = get_feature(week04['duration'],'count')
    week04_conv_dur = get_feature(week04['duration'],'sum')

    week05_conv_freq = get_feature(week05['duration'],'count')
    week05_conv_dur = get_feature(week05['duration'],'sum')

    week06_conv_freq = get_feature(week06['duration'], 'count')
    week06_conv_dur = get_feature(week06['duration'], 'sum')

    week07_conv_freq = get_feature(week07['duration'], 'count')
    week07_conv_dur = get_feature(week07['duration'], 'sum')

    week08_conv_freq = get_feature(week08['duration'], 'count')
    week08_conv_dur = get_feature(week08['duration'], 'sum')

    list_feature = all_conv_freq + all_conv_dur + week01_conv_freq + week01_conv_dur + week02_conv_freq + week02_conv_dur + \
                   week03_conv_freq + week03_conv_dur + week04_conv_freq + week04_conv_dur + week05_conv_freq + week05_conv_dur + \
                   week06_conv_freq + week06_conv_dur + week07_conv_freq + week07_conv_dur + week08_conv_freq + week08_conv_dur
    df_all.index.name = 'uid'
    df_all.loc[uid] = list_feature

#carrega o df do phq9
df_phq9=getPhq9()
df_conv_phq9 = pd.merge(df_all, df_phq9, how='inner',on='uid')

#df contendo as correlações
df_crr = pd.DataFrame(data=None, index=coll_agreg_feature, columns=['pearson_base', 'p_value_base', 'pearson_follow', 'p_value_follow'])


def creat_corr(df, index):
    pearson_base=[]
    pearson_follow = []
    p_base=[]
    p_follow=[]
    for x in coll_feature:
        pearson_coef, p_value = stats.pearsonr(df_conv_phq9['phq9','final_score_x'], df[x])
        pearson_base.append(pearson_coef)
        p_base.append(p_value)
        pearson_coef, p_value = stats.pearsonr(df_conv_phq9['phq9', 'final_score_y'], df[x])
        pearson_follow.append(pearson_coef)
        p_follow.append(p_value)
    df_crr.loc[index, 'pearson_base']= np.array(pearson_base)
    df_crr.loc[index, 'pearson_follow'] = np.array(pearson_follow)
    df_crr.loc[index, 'p_value_base'] = np.array(p_base)
    df_crr.loc[index, 'p_value_follow'] = np.array(p_follow)
#print(df_bt_phq9)

for x in coll_agreg:
    creat_corr(df_conv_phq9[x],x)
    #print(df_crr)

df_crr = df_crr.T
df_crr = df_crr.loc[:, pd.IndexSlice[:, ['mean', 'median', 'max', 'std', 'var', 'mad']]].T.reset_index()

for row, col in df_crr.iterrows():
        if 'freq' in df_crr.loc[row]['level_0']:
            df_crr.drop(index=row, inplace=True)

df_crr.reset_index()


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

layout = go.Layout(title = 'Sociabilidade - Duração de Conversação (Acompanhamento)',
                   titlefont = {'family': 'Arial',
                                'size': 22,
                                'color': '#7f7f7f'},
                   xaxis = {'title': 'Período'},
                    yaxis=dict(
                            title='Coeficiente de Correlação (r)',
                            autorange=True,
                            showgrid=False,
                            zeroline=False,
                            #range=[0.2, -0.65]

                        ),
                   paper_bgcolor = 'rgb(243, 243, 243)',
                   plot_bgcolor = 'rgb(243, 243, 243)')
fig = go.Figure(data=data, layout=layout)


py.plot(fig)


#df['Data'] = df['Data'].apply(lambda x: x.__format__('%d/%m/%Y'))
#df_conversation.drop (['start_time'], axis = 1, inplace = True)








