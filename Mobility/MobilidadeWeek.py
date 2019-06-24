from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import folium
from Scripts.PHQ9 import getPhq9
from sklearn.cluster import DBSCAN
from math import sin, cos, sqrt, atan2, radians
import plotly.offline as py
import plotly.graph_objs as go
from util.data.data_aux import get_feature_loc,join_df
from scipy import stats
user_id = ['u00','u01','u02','u03','u04','u05','u15','u25','u08','u09','u10','u12','u13','u14','u16','u17','u18','u19','u20','u22','u23',
'u27','u30','u31','u32','u33','u34','u35','u36','u41','u42','u43','u44','u45','u46','u51','u52','u57','u58','u59'
]
#'u33','u39''u43','u44''u54', , home stay
#,,'u15','u25'   'u03'

##'u00','u01','u02','u03','u04','u05','u15','u25','u08','u09','u10','u12','u13','u14','u16','u17','u18','u19','u20','u22','u23',
#'u27','u30','u31','u32','u33','u34','u35','u36','u41','u42','u43','u44','u45','u46','u51','u52','u57','u58','u59'
#'u08'
user_clt = []

coll_agreg = ['all','week01','week02','week03','week04', 'week05','week06']#,'week07']#,'week08']
coll_features=['numberOfClusters', 'totalDistance','maxDistance','maxDistanceHome','variance','standardDeviation','homeStay',
               'entropy', 'normalizedEntropy', 'transitionTime']
coll_agreg_feature = pd.MultiIndex.from_product([coll_agreg, coll_features])

coll_week = ['ALL', 'Week01', 'Week02', 'Week03', 'Week04','Week05','Week06']#,'Week07']#,'Week08']

pearson = 'pearson_follow'

def convert_date(df, columns=['timestamp']):
    for col in columns:
        df[col] = pd.to_datetime(df[col], unit='s', utc=True).dt.tz_convert(
            'US/Eastern')
    return df

def read_data(name):
    conversation_data = pd.read_csv('../SensingData/gps/gps_'+name+'.csv',index_col=False)
    return conversation_data

df_all = pd.DataFrame(data=None, index=user_id, columns=coll_agreg_feature)

for uid in user_clt:
    gps_data= read_data(uid)
    gps_data['date'] = gps_data['time'].apply(lambda x: datetime.fromtimestamp(x).date())
    gps_data = convert_date(gps_data, ['time'])
    gps_data.index = gps_data['time']
    gps_data['week_day'] = gps_data.index.dayofweek

    # remove as amostras em movimento
    #gps_data = gps_data[gps_data['travelstate'] != 'moving']
    gps_data['week_group'] = gps_data['time'].dt.to_period('W-THU')
    gps_data = gps_data[(gps_data['week_day']<5)]


    #saber quais dias considerar
    #all = gps_data.groupby(gps_data.index.date)
    #all = all[all['travelstate']!=0]

    all = gps_data

    week01 = pd.concat(
        join_df('2013-03-22/2013-03-28', '2013-03-29/2013-04-04', '2013-04-05/2013-04-11', gps_data))
    week02 = pd.concat(join_df('2013-04-05/2013-04-11', '2013-04-12/2013-04-18', None, gps_data))
    week03 = pd.concat(join_df('2013-04-12/2013-04-18', '2013-04-19/2013-04-25', None, gps_data))
    week04 = pd.concat(join_df('2013-04-19/2013-04-25', '2013-04-26/2013-05-02', None, gps_data))
    week05 = pd.concat(join_df('2013-04-26/2013-05-02', '2013-05-03/2013-05-09', None, gps_data))
    week06 = pd.concat(join_df('2013-05-03/2013-05-09', '2013-05-10/2013-05-16', None, gps_data))
    #week07 = pd.concat(join_df('2013-05-10/2013-05-16', '2013-05-17/2013-05-23', None, gps_data))
    #week08 = pd.concat(join_df('2013-05-17/2013-05-23', '2013-05-24/2013-05-30', None, gps_data))


    all_features = get_feature_loc(all)
    week01_features = get_feature_loc(week01)
    week02_features = get_feature_loc(week02)
    week03_features = get_feature_loc(week03)
    week04_features = get_feature_loc(week04)
    week05_features = get_feature_loc(week05)
    week06_features = get_feature_loc(week06)
    #week07_features = get_feature_loc(week07)
    #week08_features = get_feature_loc(week08)

    list_features = all_features + week01_features + week02_features + week03_features +\
                    week04_features + week05_features + week06_features #+ week07_features #+ week08_features

    df_all.index.name = 'uid'
    df_all.loc[uid] = list_features
    print(uid)


df_all.to_csv('mobility_week.csv')

df_all = pd.read_csv('../SensingData/mobility_week.csv',header=[0,1], index_col=[0])



#carrega o df do phq9
df_phq9=getPhq9()
df_mob_phq9 = pd.merge(df_all, df_phq9, how='inner',on='uid')

#df_feature = df_mob_phq9['all']
#df_feature['label'] = df_mob_phq9['phq9','label']

#df_feature.to_csv('feature_phq9_week.csv')


#df_mob_phq9[].to_csv('feature_phq9.csv')

#df_mob_phq9[['all','phq9']].to_csv('feaute_gps_phq9.csv')

#df contendo as correlações
df_crr = pd.DataFrame(data=None, index=coll_agreg_feature, columns=['pearson_base', 'p_value_base', 'pearson_follow', 'p_value_follow'])


def creat_corr(df, index):
    pearson_base=[]
    pearson_follow = []
    p_base=[]
    p_follow=[]
    for x in coll_features:
        pearson_coef, p_value = stats.pearsonr(df_mob_phq9['phq9','final_score_x'], df[x])
        pearson_base.append(pearson_coef)
        p_base.append(p_value)
        pearson_coef, p_value = stats.pearsonr(df_mob_phq9['phq9', 'final_score_y'], df[x])
        pearson_follow.append(pearson_coef)
        p_follow.append(p_value)
    df_crr.loc[index, 'pearson_base']= np.array(pearson_base)
    df_crr.loc[index, 'pearson_follow'] = np.array(pearson_follow)
    df_crr.loc[index, 'p_value_base'] = np.array(p_base)
    df_crr.loc[index, 'p_value_follow'] = np.array(p_follow)

for x in coll_agreg:
    creat_corr(df_mob_phq9[x],x)

#print(df_crr[:,'numberOfClusters'])
#print(df_crr[df_crr['pearson_follow'] < -0.3]['pearson_follow'])


df_crr = df_crr.T
df_crr = df_crr.loc[:, pd.IndexSlice[:, ['numberOfClusters', 'totalDistance','maxDistance',
                'maxDistanceHome','variance','standardDeviation','homeStay',
               'entropy', 'normalizedEntropy', 'transitionTime']]].T.reset_index()

#df_crr['semana']  = coll_week
#print(df_crr[df_crr['level_1']=='numberOfClusters']['pearson_follow'])

df_crr_final = df_crr[df_crr['level_0'] == 'all'][['pearson_base','pearson_follow','p_value_base','p_value_follow']]
df_crr_final.index = coll_features





def create_chart(statist='mean', pearson='pearson_base', name=None, color = '#341f97'):
    df_crr_var = df_crr[df_crr['level_1']==statist]
    df_crr_var['semana'] = coll_week
    # Gráfico usando apenas marcadores
    trace = go.Scatter(x=df_crr_var['semana'],
                        y=df_crr_var[pearson],
                        mode='markers+lines',
                        name=name,
                        line={'color': color},
                        )
    return trace

pearson = 'pearson_follow'

trace1 = create_chart('numberOfClusters',pearson, 'numberOfClusters', '#ee5253' )
#trace2 = create_chart('totalDistance',pearson, 'totalDistance', '#009900')
trace3 = create_chart('maxDistance',pearson, 'maxDistance', '#341f97')
#trace4 = create_chart('maxDistanceHome',pearson, 'maxDistanceHome', '#8B4513')
#trace5 = create_chart('variance',pearson, 'variance', '#8B008B')
#trace6 = create_chart('standardDeviation',pearson, 'standardDeviation', '#800000')

trace7 = create_chart('homeStay',pearson, 'homeStay', '#248eff')
trace8 = create_chart('entropy',pearson, 'entropy', '#7b8a24')
trace9 = create_chart('normalizedEntropy',pearson, 'normalizedEntropy', '#000000')
#trace10 = create_chart('transitionTime',pearson, 'transitionTime', '#8B8B83')



data = [trace1,trace3,trace7, trace8,trace9]

layout = go.Layout(title = 'Mobilidade Semana (Acompanhamento)',
                   titlefont = {'family': 'Arial',
                                'size': 22,
                                'color': '#7f7f7f'},
                   xaxis = {'title': 'Período'},


                    yaxis=dict(
                            title='Coeficiente de Correlação (r)',
                            autorange=True,
                            showgrid=False,
                            zeroline=False,
                            #range=[0.6, -0.65]

                        ))
                   #paper_bgcolor = 'rgb(243, 243, 243)',
                   #plot_bgcolor = 'rgb(243, 243, 243)')

fig = go.Figure(data=data, layout=layout)


#py.plot(fig)







