import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
from scipy import stats
from Scripts.PHQ9 import getPhq9
import plotly.offline as py
import plotly.graph_objs as go

#remover frequencia zerada

user_id = ['u00','u01','u02','u03','u04','u05','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22',
        'u23','u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44','u45'
    ,'u46','u51','u52','u54','u57','u58','u59']

coll_week = ['ALL', 'Week01', 'Week02', 'Week03', 'Week04','Week05','Week06','Week07','Week08']

#nan -> 'u24','u07',     ,'u47','u49','u50','u53'  ,'u56'

#user_id = ['u01','u02','u03','u04','u10','u12','u14','u15','u16','u17','u18','u19','u20','u22',
 #       'u23','u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43']
#,'u24' está com nan

#user_id = ['u03']

#col_week=['all_freq_day_mean','all_uni_day_mean','week01_freq_day_mean','week01_uni_day_mean', 'week02_freq_day_mean','week02_uni_day_mean','week03_freq_day_mean','week03_uni_day_mean',
 #   'week04_freq_day_mean','week04_uni_day_mean','week05_freq_day_mean','week05_uni_day_mean','week06_freq_day_mean','week06_uni_day_mean',
  #  'week07_freq_day_mean','week07_uni_day_mean','week08_freq_day_mean','week08_uni_day_mean']

coll_agreg = ['all_freq_day', 'all_uni_day', 'week01_freq_day','week01_uni_day','week02_freq_day','week02_uni_day',
                'week03_freq_day', 'week03_uni_day','week04_freq_day','week04_uni_day','week05_freq_day','week05_uni_day',
                'week06_freq_day', 'week06_uni_day','week07_freq_day','week07_uni_day','week08_freq_day','week08_uni_day']
coll_feature=['mean', 'median', 'max', 'std', 'var', 'mad']
coll_agreg_feature = pd.MultiIndex.from_product([coll_agreg, coll_feature])

def convert_date(df, columns=['timestamp']):
    for col in columns:
        df[col] = pd.to_datetime(df[col], unit='s', utc=True).dt.tz_convert(
            'US/Eastern')
    return df


def read_data(name):
    conversation_data = pd.read_csv('../SensingData/bluetooth/bt_'+name+'.csv')
    return conversation_data


def get_feature(data,group='none'):
    if group == 'count':
        data = data.count()
    mean = data.mean()
    median = data.median()
    max = data.max()
    std = data.std()
    var = data.var()
    mad = data.mad()
    return [mean, median, max, std, var, mad]

df_all = pd.DataFrame(data=None, index=user_id, columns=coll_agreg_feature)

def join_df(date1,date2,date3):
    aux1 = bt_data[bt_data['week_group'] == date1]
    aux2 = bt_data[bt_data['week_group'] == date2]
    if date3 == "":
        join = [aux1, aux2]
    else:
        aux3 = bt_data[bt_data['week_group'] == date3]
        join = [aux1,aux2,aux3]
    return join

    #def clear_data(data):
     #   count = data['MAC'].count()
      #  return data[data>0]

def ent(data):
    p_data= data.value_counts()/len(data) # calculates the probabilities
    entropy=sc.stats.entropy(p_data)  # input probabilities to get the entropy
    return entropy

for uid in user_id:
    bt_data = read_data(uid)
    bt_data.drop('class_id', inplace=True, axis=1)
    convert_date(bt_data, columns=['time'])
    bt_data.index = bt_data['time']
    bt_data['week_day'] = bt_data['time'].apply(lambda x: x.weekday())
    bt_data['week_group'] = bt_data['time'].dt.to_period('W-THU')

    all = bt_data.resample('D')

    week01 = pd.concat(join_df('2013-03-22/2013-03-28','2013-03-29/2013-04-04','2013-04-05/2013-04-11')).resample('D')
    week02 = pd.concat(join_df('2013-04-05/2013-04-11', '2013-04-12/2013-04-18','')).resample('D')
    week03 = pd.concat(join_df('2013-04-12/2013-04-18','2013-04-19/2013-04-25','')).resample('D')
    week04 = pd.concat(join_df('2013-04-19/2013-04-25','2013-04-26/2013-05-02','')).resample('D')
    week05 = pd.concat(join_df('2013-04-26/2013-05-02', '2013-05-03/2013-05-09','')).resample('D')
    week06 = pd.concat(join_df('2013-05-03/2013-05-09','2013-05-10/2013-05-16','')).resample('D')
    week07 = pd.concat(join_df('2013-05-10/2013-05-16','2013-05-17/2013-05-23','')).resample('D')
    week08 = pd.concat(join_df('2013-05-17/2013-05-23','2013-05-24/2013-05-30','')).resample('D')

#tratar dias zerados
    #print(all['MAC'].count().replace(0,np.nan))
    #teste = all['MAC'].count()
    #print(teste[teste>0])

    all_bt_freq = get_feature(all['MAC'],'count')
    all_bt_uni =  get_feature(all['MAC'].apply(lambda x: x.nunique()))

    week01_bt_freq = get_feature(week01['MAC'],'count')
    week01_bt_uni = get_feature(week01['MAC'].apply(lambda x: x.nunique()))

    week02_bt_freq = get_feature(week02['MAC'],'count')
    week02_bt_uni = get_feature(week02['MAC'].apply(lambda x: x.nunique()))

    week03_bt_freq = get_feature(week03['MAC'],'count')
    week03_bt_uni = get_feature(week03['MAC'].apply(lambda x: x.nunique()))

    week04_bt_freq = get_feature(week04['MAC'],'count')
    week04_bt_uni = get_feature(week04['MAC'].apply(lambda x: x.nunique()))

    week05_bt_freq = get_feature(week05['MAC'],'count')
    week05_bt_uni = get_feature(week05['MAC'].apply(lambda x: x.nunique()))

    week06_bt_freq = get_feature(week06['MAC'],'count')
    week06_bt_uni = get_feature(week06['MAC'].apply(lambda x: x.nunique()))

    week07_bt_freq = get_feature(week07['MAC'],'count')
    week07_bt_uni = get_feature(week07['MAC'].apply(lambda x: x.nunique()))

    week08_bt_freq = get_feature(week08['MAC'],'count')
    week08_bt_uni = get_feature(week08['MAC'].apply(lambda x: x.nunique()))

    list_feature = all_bt_freq+all_bt_uni+week01_bt_freq+ week01_bt_uni+ week02_bt_freq+ week02_bt_uni+\
                   week03_bt_freq+ week03_bt_uni+ week04_bt_freq+ week04_bt_uni+ week05_bt_freq+ week05_bt_uni+ \
                   week06_bt_freq+ week06_bt_uni+ week07_bt_freq+ week07_bt_uni+ week08_bt_freq+ week08_bt_uni
    df_all.index.name = 'uid'
    df_all.loc[uid] = list_feature

#df_phq9 = pd.read_csv('../DadosLimpos/LonelinessScale.csv')
#df_phq9 = pd.read_csv('../DadosLimpos/phq9.csv')
df_phq9=getPhq9()

df_bt_phq9 = pd.merge(df_all, df_phq9, how='inner',on='uid')

#columns_corr = pd.MultiIndex.from_product(['phq9', ['pearson_base', 'p_value_base', 'pearson_follow', 'p_value_follow']])

df_crr = pd.DataFrame(data=None, index=coll_agreg_feature, columns=['pearson_base', 'p_value_base', 'pearson_follow', 'p_value_follow'])

def creat_corr(df, index):
    pearson_base=[]
    pearson_follow = []
    p_base=[]
    p_follow=[]
    for x in coll_feature:
        pearson_coef, p_value = stats.pearsonr(df_bt_phq9['phq9','final_score_x'], df[x])
        pearson_base.append(pearson_coef)
        p_base.append(p_value)
        pearson_coef, p_value = stats.pearsonr(df_bt_phq9['phq9', 'final_score_y'], df[x])
        pearson_follow.append(pearson_coef)
        p_follow.append(p_value)
    df_crr.loc[index,'pearson_base'] = pearson_base
    df_crr.loc[index, 'pearson_follow'] = pearson_follow
    df_crr.loc[index, 'p_value_base'] = p_base
    df_crr.loc[index, 'p_value_follow'] = p_follow
#print(df_bt_phq9)



#df_crr.loc['all_freq_day','p_value']=[1,2,3,4,5,6]
for x in coll_agreg:
    creat_corr(df_bt_phq9[x],x)

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

trace1 = create_chart('median','pearson_base', 'Mediana', '#ee5253' )
trace2 = create_chart('mean','pearson_base', 'Média', '#009900')
trace3 = create_chart('var','pearson_base', 'Variância', '#341f97')
trace4 = create_chart('max','pearson_base', 'Máximo', '#8B4513')
trace5 = create_chart('std','pearson_base', 'STD', '#8B008B')
trace6 = create_chart('mad','pearson_base', 'Mad', '#800000')
data = [trace1, trace2,trace3, trace4, trace5, trace6]

layout = go.Layout(title = 'Sociabilidade - Co-localizações Distintas (Linha de Base)',
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


