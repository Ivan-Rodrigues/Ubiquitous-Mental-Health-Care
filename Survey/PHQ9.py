import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

user_id = ['u00','u01','u02','u03','u04','u05','u07','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22',
        'u23','u24','u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44',
         'u45','u46','u47','u49','u50','u51','u52','u53','u54','u56','u57','u58','u59']

phq9 = pd.read_csv('../Surveys/PHQ-9.csv')
phq9.columns = ['uid','type','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Response']

score = {'Not at all': 0, 'Several days': 1, 'More than half the days': 2, 'Nearly every day':3}
question_cols = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9']

for q in question_cols:
    phq9[q] = phq9[q].apply(lambda x:score.get(x))

phq9['final_score'] = phq9[question_cols].apply(np.sum, axis=1)

def calc_score(x):
    if x <=4:
        return 'minimal'
    elif x <= 9:
        return 'mild'
    elif x <= 14:
        return 'moderate'
    elif x <= 19:
        return 'moderately_severe'
    else:
        return 'severe'

def calc_label(x):
    if x <=5:
        return 0
    else:
        return 1

"""

def calc_score(x):
    if x <5:
        return 'non-depressive'
    elif x>=9 and x<15:
        return 'moderate'
    else:
        return 'depressive'
"""
phq9['depression_level'] = phq9['final_score'].apply(lambda x: calc_score(x))

phq9['label'] = phq9['final_score'].apply(lambda x: calc_label(x))


pre_final = phq9[phq9.type == 'pre'][['uid','final_score','depression_level']]
post_final = phq9[phq9.type == 'post'][['uid','final_score','depression_level', 'label']]

merge_phq9 = pd.merge(pd.DataFrame(pre_final), pd.DataFrame(post_final), how='inner', on='uid')



merge_phq9.index = merge_phq9['uid']
merge_phq9.drop(inplace=True, columns=['uid'])
columns = pd.MultiIndex.from_product([['phq9'],
                                      ['final_score_x', 'depression_level_x', 'final_score_y','depression_level_y', 'label']])


df_all_phq9 = pd.DataFrame(data=None, index=merge_phq9.index, columns=columns)
df_all_phq9['phq9'] = merge_phq9


def getPhq9():
    return df_all_phq9


#df_all_phq9.to_csv("phq9.csv")

#sns.catplot(x="label", kind="count", data=df_all_phq9['phq9'])

#groupPlot = pre_post_merge.iloc[1:10]


#sns.distplot(pre_post_merge['final_score_y'], kde=False)

print(plt.show())





