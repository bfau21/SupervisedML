import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import pretty_errors
from sklearn.svm import SVC



features = pd.read_csv('alt_acsincome_ca_features_85.csv')
labels = pd.read_csv('alt_acsincome_ca_labels_85.csv')

# Préparation des données
data = features.copy()
data['PINCP'] = labels

# Définition de X et y
X, y = features, labels['PINCP'].ravel()

print(X.columns[8])