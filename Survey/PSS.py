import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pss = pd.read_csv('../Surveys/PerceivedStressScale.csv')
pss.columns = ['uid','type','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10']
score = {'Never': 0, 'Almost never': 1, 'Sometime': 2, 'Fairly often':3, 'Very often':4}
question_cols = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10']

#Tem algumas questões nulas

for q in question_cols:
    pss[q] = pss[q].apply(lambda x:score.get(x))

pss['final_score'] = pss[question_cols].apply(np.sum, axis=1)

def calc_score(x):
    if x <=13:
        return 'low'
    elif x <= 26:
        return 'moderate'
    else:
        return 'high'
pss['stress_level'] = pss['final_score'].apply(lambda x: calc_score(x))

pre_final = pss[pss.type == 'pre'][['uid','final_score','stress_level']]
post_final = pss[pss.type == 'post'][['uid','final_score','stress_level']]

merge_pss = pd.merge(pd.DataFrame(pre_final), pd.DataFrame(post_final), how='inner',on='uid')



sns.distplot(merge_pss['final_score_x'], kde=False, bins=22)
sns.distplot(merge_pss['final_score_y'], kde=False, bins=22)

print(plt.show())
