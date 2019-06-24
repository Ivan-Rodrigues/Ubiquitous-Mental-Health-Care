import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

loneliness = pd.read_csv('../Surveys/LonelinessScale.csv')

loneliness.columns = ['uid','type','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12',
            'Q13', 'Q14', 'Q15', 'Q16','Q17','Q18','Q19','Q20']

score = {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often':4}
question_cols = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12',
            'Q13', 'Q14', 'Q15', 'Q16','Q17','Q18','Q19','Q20']

for q in question_cols:
    loneliness[q] = loneliness[q].apply(lambda x:score.get(x))

loneliness['final_score'] = loneliness[question_cols].apply(np.sum, axis=1)

def calc_score(x):
    if x <=50:
        return 'low'
    elif x <= 59:
        return 'moderate'
    else:
        return 'high'

loneliness['level'] = loneliness['final_score'].apply(lambda x : calc_score(x))

pre_final = loneliness[loneliness.type == 'pre'][['uid','final_score','level']]
post_final = loneliness[loneliness.type == 'post'][['uid','final_score','level']]

merge = pd.merge(pd.DataFrame(pre_final), pd.DataFrame(post_final), how='inner', on='uid')

def getLoneliness():
    return merge

#sns.distplot(merge['final_score_x'],kde=False, bins=30)
#sns.distplot(merge['final_score_y'], kde=False, bins=30)

#print(plt.show())
