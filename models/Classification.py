import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,LeaveOneOut,cross_val_score,ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


df_all = pd.read_csv('../../SensingData/feature_phq9_all.csv')
df_week = pd.read_csv('../../SensingData/feature_phq9_week.csv')
df_weekend = pd.read_csv('../../SensingData/feature_phq9_weekend.csv')


list_features = ['numberOfClusters','homeStay','entropy', 'normalizedEntropy', 'maxDistance']

def classitication_depression(features, labels, clasify = 'rf'):

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.30, random_state = 42)

    # Create a random forest Classifier. By convention, clf means 'Classifier'
    clf_rf = RandomForestClassifier(n_jobs=2, random_state=0)

    # Create a svm Classifier
    clf_svm = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the Classifier to take the training features and learn how they relate
    # to the training y (the species)

    if clasify == 'rf':
        clf_rf.fit(train_features, train_labels)
        clf = clf_rf
    else:
        clf_svm.fit(train_features, train_labels)
        clf = clf_svm

    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)
    scores_acuracy = cross_val_score(clf , X = features , y = labels, cv=cv, scoring='accuracy')
    scores_precision = cross_val_score(clf , X = features , y = labels, cv=cv, scoring='precision')
    scores_recal = cross_val_score(clf , X = features , y = labels, cv=cv, scoring='recall')
    scores_f1score = cross_val_score(clf , X = features , y = labels, cv=cv, scoring='f1')

    print('\n Avaliação para as 10 Semanas')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores_acuracy.mean(), scores_acuracy.std() * 2))
    print("Precision: %0.2f (+/- %0.2f)" % (scores_precision.mean(), scores_precision.std() * 2))
    print("Recal: %0.2f (+/- %0.2f)" % (scores_recal.mean(), scores_recal.std() * 2))
    print("F1 Score: %0.2f (+/- %0.2f)" % (scores_f1score.mean(), scores_f1score.std() * 2))
    print(" ----------------------------------------------- \n")

# modelo de classificação usando as 9 semanas
features_all = df_all[list_features]
labels = df_all['label']
classitication_depression(features_all, labels, 'rf')

#Modelo de classificação usando características dos dias semana
features_week = df_week[list_features]
labels = df_week['label']
classitication_depression(features_week, labels, 'rf')

#Modelo de classificação usando características dos finais de semana
features_weekend = df_weekend[list_features]
labels = df_weekend['label']
classitication_depression(features_weekend, labels, 'rf')






