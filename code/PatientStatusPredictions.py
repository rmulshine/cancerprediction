#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 15:07:40 2021

@author: rileymulshine
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

import time

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

from scipy.stats import binom

cancer_data = pd.read_csv('finalcancerdata.csv')

# Adjust Cancer_type into dummy variables
cancer_types = pd.get_dummies(cancer_data.Cancer_type)
cancer_data = pd.concat([cancer_data, pd.DataFrame(cancer_types)], axis = 1)


# NAIVE BAYES ALGORITHM -----------------------------------

Y = cancer_data.Status_patient

features = ['Age', 'MStatus', 'Edu_status', 'Family_history', 'Alcohol', 'Tobacco', 'Khat', 'Biopsy', 'US', 'Hist_type', 'Hist_grade', 'TNM_stage', 'Chemotherapy', 'Radiotherapy', 'colorectal', 'esophageal', 'prostate']
features_meta = ['Age', 'MStatus', 'Edu_status', 'Family_history', 'Alcohol', 'Tobacco', 'Khat', 'Biopsy', 'US', 'Hist_type', 'Hist_grade', 'TNM_stage', 'Dist_metastasis', 'Chemotherapy', 'Radiotherapy', 'colorectal', 'esophageal', 'prostate']

# No metastasis data
X = cancer_data[features]

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.5, random_state = 0)
# Choosing train-test split as 50-50, because the aim of the study is to investigate the accuracy of the test (greater number of test trials is optimal)

gnb = GaussianNB()
Y_pred = gnb.fit(train_X, train_Y).predict(test_X)

total_preds = test_X.shape[0]
correct_no_meta = (test_Y == Y_pred).sum()
accuracy_no_meta = accuracy_score(test_Y, Y_pred)

# With metastasis data
gnb_meta = GaussianNB()
X_meta = cancer_data[features_meta]

train_X_meta, test_X_meta, train_Y_meta, test_Y_meta = train_test_split(X_meta, Y, test_size = 0.5, random_state = 0)

Y_pred_meta = gnb_meta.fit(train_X_meta, train_Y_meta).predict(test_X_meta)

correct_meta = (test_Y_meta == Y_pred_meta).sum()
accuracy_meta = accuracy_score(test_Y_meta, Y_pred_meta)
meta_improvement_nb = (accuracy_meta - accuracy_no_meta)/accuracy_no_meta

mat_nb_no = confusion_matrix(test_Y, Y_pred)
mat_nb_meta = confusion_matrix(test_Y_meta, Y_pred_meta)
fig, (ax1, ax2) = plt.subplots(1,2)
sns.heatmap(mat_nb_no.T, square=True, annot=True, fmt='d', cbar=False, xticklabels = ["Survival", "Decease"], yticklabels = ["Survival", "Decease"], ax = ax1)
sns.heatmap(mat_nb_meta.T, square=True, annot=True, fmt='d', cbar=False, xticklabels = ["Survival", "Decease"], yticklabels = ["Survival", "Decease"], ax = ax2)
plt.show()



# RANDOM FOREST ALGORITHM -----------------------------------

# No metastasis data

def get_mae(nestimates, train_x, test_x, train_y, test_y):
    model = RandomForestClassifier(n_estimators = nestimates, random_state = 0)
    model.fit(train_x, train_y)
    preds_test = model.predict(test_x)
    mae = mean_absolute_error(test_y, preds_test)
    return(mae)

n_estimates_array = [100000000, 1]
for nestimates in [5, 50, 100, 500, 1000]:
    my_mae = get_mae(nestimates, train_X, test_X, train_Y, test_Y)
    if my_mae < n_estimates_array[0]:
        n_estimates_array[0] = my_mae
        n_estimates_array[1] = nestimates

forest = RandomForestClassifier(n_estimators = n_estimates_array[1], random_state = 0)
forest.fit(train_X, train_Y)
preds_test = forest.predict(test_X)
preds_test = np.around(preds_test)

correct_forest_no = (test_Y == preds_test).sum()
accuracy_forest_no = accuracy_score(test_Y, preds_test)

#With metastasis data

n_estimates_meta = [100000000, 1]
for nestimates in [5, 50, 100, 500, 1000]:
    my_mae = get_mae(nestimates, train_X, test_X, train_Y, test_Y)
    if my_mae < n_estimates_meta[0]:
        n_estimates_meta[0] = my_mae
        n_estimates_meta[1] = nestimates

forest_meta = RandomForestClassifier(n_estimators = n_estimates_meta[1], random_state = 0)
forest_meta.fit(train_X_meta, train_Y_meta)
preds_test_meta = forest_meta.predict(test_X_meta)
preds_test_meta = np.around(preds_test_meta)

correct_meta_forest = (test_Y_meta == preds_test_meta).sum()
accuracy_meta_forest = accuracy_score(test_Y_meta, preds_test_meta)
meta_improvement_forest = (accuracy_meta_forest - accuracy_forest_no)/accuracy_forest_no

mat_forest_no = confusion_matrix(test_Y, preds_test)
mat_forest_meta = confusion_matrix(test_Y_meta, preds_test_meta)
fig, (ax1, ax2) = plt.subplots(1,2)
sns.heatmap(mat_forest_no.T, square=True, annot=True, fmt='d', cbar=False, xticklabels = ["Survival", "Decease"], yticklabels = ["Survival", "Decease"], ax = ax1)
sns.heatmap(mat_forest_meta.T, square=True, annot=True, fmt='d', cbar=False, xticklabels = ["Survival", "Decease"], yticklabels = ["Survival", "Decease"], ax = ax2)
plt.show()


start_time = time.time()
result = permutation_importance(
    forest_meta, test_X_meta, test_Y_meta, n_repeats=10, random_state=0, n_jobs=2)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: "
      f"{elapsed_time:.3f} seconds")
forest_importances = pd.Series(result.importances_mean, index=features_meta)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

fn = features_meta
cn = "Status_patient"
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
plot_tree(forest_meta.estimators_[0], feature_names = fn, class_names=cn, filled = True);

# XGBOOST ALGORITHM -----------------------------------

# No metastasis data
xgb_model_no = XGBClassifier()
xgb_model_no.fit(train_X, train_Y)
preds_xgb_no = xgb_model_no.predict(test_X)
preds_xgb_no = np.around(preds_xgb_no)

correct_xgb_no = (test_Y == preds_xgb_no).sum()
accuracy_xgb_no = accuracy_score(test_Y, preds_xgb_no)

# With metastasis data
xgb_model_meta = XGBClassifier()
xgb_model_meta.fit(train_X_meta, train_Y_meta)
preds_xgb_meta = xgb_model_meta.predict(test_X_meta)
preds_xgb_meta = np.around(preds_xgb_meta)

correct_xgb_meta = (test_Y_meta == preds_xgb_meta).sum()
accuracy_xgb_meta = accuracy_score(test_Y_meta, preds_xgb_meta)

mat_xgb_no = confusion_matrix(test_Y, preds_xgb_no)
mat_xgb_meta = confusion_matrix(test_Y_meta, preds_xgb_meta)
fig, (ax1, ax2) = plt.subplots(1,2)
sns.heatmap(mat_xgb_no.T, square=True, annot=True, fmt='d', cbar=False, xticklabels = ["Survival", "Decease"], yticklabels = ["Survival", "Decease"], ax = ax1)
sns.heatmap(mat_xgb_meta.T, square=True, annot=True, fmt='d', cbar=False, xticklabels = ["Survival", "Decease"], yticklabels = ["Survival", "Decease"], ax = ax2)
plt.show()


print("The accuracy scores for the models are:")
print(f"Naive Bayes Model: without metastasis data = {round(accuracy_no_meta, 3)};     with metastasis data = {round(accuracy_meta, 3)}")
print(f"Random Forest Model: without metastasis data = {round(accuracy_forest_no, 3)};     with metastasis data = {round(accuracy_meta_forest, 3)}")
print(f"XGBoost Model: without metastasis data = {round(accuracy_xgb_no, 3)};     with metastasis data = {round(accuracy_xgb_meta, 3)}")

# ENSEMBLE MODEL --------------------------------

estimators_no = [('gnb', gnb), ('rf', forest), ('xgb', xgb_model_no)]
ensemble_no = VotingClassifier(estimators_no, voting = 'hard')
ensemble_no.fit(train_X, train_Y)
preds_ensemble_no = ensemble_no.predict(test_X)
correct_ensemble_no = (test_Y == preds_ensemble_no).sum()
accuracy_ensemble_no = accuracy_score(test_Y, preds_ensemble_no)
print(f"Accuracy ensemble without meta: {round(accuracy_ensemble_no, 3)}")

estimators_meta = [('gnb', gnb_meta), ('rf', forest_meta), ('xgb', xgb_model_meta)]
ensemble_meta = VotingClassifier(estimators_meta, voting = 'hard')
ensemble_meta.fit(train_X_meta, train_Y_meta)
preds_ensemble_meta = ensemble_meta.predict(test_X_meta)
correct_ensemble_meta = (test_Y_meta == preds_ensemble_meta).sum()
accuracy_ensemble_meta = accuracy_score(test_Y_meta, preds_ensemble_meta)
print(f"Accuracy ensemble with meta: {round(accuracy_ensemble_meta, 3)}")

# Hypothesis tests to determine if Dist_metastasis increases prediction accuracy
p_meta = 1 - binom.cdf((correct_ensemble_meta - 1), total_preds, accuracy_ensemble_no)

reject_h0 = False
if p_meta < 0.05:
    reject_h0 = True
