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


models = {
    "RandomForest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "Stacking": StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('gb', GradientBoostingClassifier(n_estimators=100))
        ],
        final_estimator=LogisticRegression()
    )
}


param_grid_rf = {
    'n_estimators': [50, 100, 200],  
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],  
    'max_features': ['auto', 'sqrt', 'log2'],  
    'bootstrap': [True, False]  
}

param_grid_ab = {
    'n_estimators': [50, 100, 200],  
    'learning_rate': [0.01, 0.1, 1.0, 10.0],  
}

param_grid_gb = {
    'n_estimators': [50, 100, 200],  
    'learning_rate': [0.01, 0.1, 0.2],  
    'max_depth': [3, 5, 7], 
    'subsample': [0.8, 0.9, 1.0], 
    'min_samples_split': [2, 5],  
}

param_grid_stack = {
    'final_estimator__C': [0.1, 1, 10], 
    'final_estimator__penalty': ['l2'], 
    'final_estimator__solver': ['lbfgs']  
}

list_param_grids = [param_grid_rf, param_grid_ab, param_grid_gb, param_grid_stack]
list_models_files = ['RandomForest_BestModel_0819.joblib', 'AdaBoost_BestModel_0810.joblib', 'GradientBoosting_BestModel_0827.joblib', 'Stacking_BestModel_0822.joblib'] #Liste obtenue après avoir fait le GridSearch

#list_models_files = []

# best_models = { 
#     "RandomForest": joblib.load(list_models_files[0]),
#     "AdaBoost": joblib.load(list_models_files[1]),
#     "GradientBoosting": joblib.load(list_models_files[2]),
#     "Stacking": joblib.load(list_models_files[3])
# } 



# Chargement des données et prétraitement
def load_and_preprocess_data(features_filepath, labels_filepath):
    X = pd.read_csv(features_filepath) # Features columns
    y = pd.read_csv(labels_filepath)['PINCP'] #Labels
    return X, y

# Analyse de la distribution des attributs
def analyse_feature_distribution(X):
    for column in X.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(X[column], kde=True, bins=40, color='red')
        plt.title(f"Distribution de l'attribut {column}")
        plt.savefig(f'distribution_{column}.png')  
        plt.close()  

# Mélange et partitionnement
def split_and_standardize(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, shuffle=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.joblib')  
    return X_train_scaled, X_test_scaled, y_train, y_test

# Évaluation des modèles par défaut
def evaluate_default_models(X_train, X_test, y_train, y_test):
    results = {}
    for model_name, model in models.items():
        print(f"Évaluation du modèle : {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy : {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        results[model_name] = accuracy
    return results


# Optimisation avec GridSearchCV
def optimize_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_

def save_best_model(model, accuracy, model_name):
    accuracy_str = str(round(accuracy, 4)).replace('.', '')[:4]
    filename = f"{model_name}_BestModel_{accuracy_str}.joblib"
    joblib.dump(model, filename)
    print(f"Modèle {model_name} enregistré dans : {filename}")

# Inférence sur de nouveaux jeux de données
def predict_new_data(scaler_file, model_file, X):
    scaler = joblib.load(scaler_file)
    model = joblib.load(model_file)
    new_features_scaled = scaler.transform(X)
    predictions = model.predict(new_features_scaled)
    return predictions

def plot_correlation_matrix(data, target_column):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Corrélations des features avec le label ({target_column})')
    plt.savefig(f'correlation_{target_column}.png')  
    plt.close()  

def plot_feature_importance(X, y, model_type='RandomForest'):
    if model_type == 'RandomForest':
        model = joblib.load("RandomForest_BestModel_0819.joblib")
    elif model_type == 'GradientBoosting':
        model = joblib.load("GradientBoosting_BestModel_0827.joblib")
    elif model_type == 'AdaBoost':
        model = joblib.load("AdaBoost_BestModel_0810.joblib")
    elif model_type == 'Stacking':
        model = joblib.load("Stacking_BestModel_0822.joblib")
    else:
        raise ValueError("Mauvais model_type")
    model.fit(X, y)
    importances = model.feature_importances_
    sorted_idx = importances.argsort()
    plt.figure(figsize=(10, 6))
    plt.barh(range(10), importances[sorted_idx[-10:]])
    plt.yticks(range(10), X.columns[sorted_idx[-10:]])
    plt.title(f'10 caractéristiques les plus importantes ({model_type})')
    plt.savefig(f'importance_{model_type}.png')  
    plt.close() 

def evaluate_gender_bias(X, y, gender_column='SEX'):
    data = X.copy()
    data['PINCP'] = y
    # Taux global
    global_rate = data['PINCP'].mean() * 100
    print(f"Taux global d'individus avec un revenu > 50 000 : {global_rate:.2f}%")
    # Vérifier si la colonne de genre existe
    if gender_column not in data.columns:
        print(f"La colonne {gender_column} n'existe pas dans les données.")
        return
    # Taux par genre
    for gender, label in zip([1, 2], ['Homme', 'Femme']):
        gender_data = data[data[gender_column] == gender]
        rate = gender_data['PINCP'].mean() * 100
        print(f"Taux pour {label} avec un revenu > 50 000 : {rate:.2f}%")

def evaluate_fairness_by_gender(model, X, y, gender_column='SEX'):
    data = X.copy()
    data['PINCP'] = y
    if gender_column not in data.columns:
        print(f"La colonne {gender_column} n'existe pas dans les données.")
        return
    # Prédictions
    predictions = model.predict(X)
    # Analyse par genre
    metrics = {}
    for gender, label in zip([1, 2], ['Homme', 'Femme']):
        gender_data = data[data[gender_column] == gender]
        gender_y_true = gender_data['PINCP']
        gender_y_pred = predictions[gender_data.index]
        # Matrice de confusion
        tn, fp, fn, tp = confusion_matrix(gender_y_true, gender_y_pred).ravel()
        # Calcul des métriques
        total = len(gender_y_true)
        ppr = (tp + fp) / total  # Positive Prediction Rate
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        metrics[label] = {
            "PPV (Statistical Parity)": ppr,
            "TPR (Equal Opportunity)": tpr,
            "FPR (Predictive Equality)": fpr,
        }
    return metrics

def calculate_fairness_metrics(y_true, y_pred, gender_column):
    metrics = {}
    for gender in ['Homme', 'Femme']:
        indices = (gender_column == gender)
        y_true_gender = y_true[indices]
        y_pred_gender = y_pred[indices]
        tn, fp, fn, tp = confusion_matrix(y_true_gender, y_pred_gender).ravel()
        
        ppv = (tp + fp) / len(y_pred_gender)  # Statistical Parity
        tpr = tp / (tp + fn)  # Equal Opportunity
        fpr = fp / (fp + tn)  # Predictive Equality
        
        metrics[gender] = {
            "Statistical Parity (PPV)": ppv,
            "Equal Opportunity (TPR)": tpr,
            "Predictive Equality (FPR)": fpr
        }
    return metrics









def main():
    # Load California data
    X, y = load_and_preprocess_data('alt_acsincome_ca_features_85.csv','alt_acsincome_ca_labels_85.csv')
    # Analyze features distribution
    analyse_feature_distribution(X)
    # Split & Standardize
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_standardize(X, y)
    # Evaluate default models
    results = evaluate_default_models(X_train_scaled, X_test_scaled, y_train, y_test)
    print(results)
    # Search & Save best models
    rf_best_n_estimator = -1
    gb_best_n_estimator = -1
    for i, model_items in enumerate(list(models.items())[0:3]): #RF, AB, GB
        model_name, model = model_items
        best_model, best_score, best_params = optimize_model(model, list_param_grids[i], X_train_scaled, y_train)
        if i == 0: #Random Forest
            rf_best_n_estimator = best_model.n_estimators
        elif i == 2: #Gradient Boosting
            gb_best_n_estimator = best_model.n_estimators
        save_best_model(best_model, best_score, model_name)
        accuracy_str = str(round(best_score, 4)).replace('.', '')[:4]
        list_models_files.append(f"{model_name}_BestModel_{accuracy_str}.joblib")
    # Stacking
    best_model, best_score, best_params = optimize_model(StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=rf_best_n_estimator)),
            ('gb', GradientBoostingClassifier(n_estimators=gb_best_n_estimator))
        ],
        final_estimator=LogisticRegression()
    ), list_param_grids[3], X_train_scaled, y_train)
    save_best_model(best_model, best_score, 'Stacking')
    accuracy_str = str(round(best_score, 4)).replace('.', '')[:4]
    list_models_files.append(f"Stacking_BestModel_{accuracy_str}.joblib")
    # Test Nevada
    best_models = { 
        "RandomForest": joblib.load(list_models_files[0]),
        "AdaBoost": joblib.load(list_models_files[1]),
        "GradientBoosting": joblib.load(list_models_files[2]),
        "Stacking": joblib.load(list_models_files[3])
    } 
    X, y = load_and_preprocess_data('acsincome_ne_features.csv','acsincome_ne_labelTP2.csv')
    #X, y = load_and_preprocess_data('features_split_1.csv', 'labels_split_1.csv')
    for i, model_items in enumerate(best_models.items()):
        model_name, model = model_items
        results = {}
        nevada_pred = predict_new_data('scaler.joblib', list_models_files[i], X)
        accuracy = accuracy_score(y, nevada_pred)
        print(f"Accuracy : {accuracy:.4f}")
        print(classification_report(y, nevada_pred))
        print(confusion_matrix(y, nevada_pred))
        results[model_name] = accuracy
        print(results)
    # Test Colorado
    X, y = load_and_preprocess_data('acsincome_co_features.csv','acsincome_co_label.csv')
    for i, model_items in enumerate(best_models.items()):
        model_name, model = model_items
        results = {}
        colorado_pred = predict_new_data('scaler.joblib', list_models_files[i], X)
        accuracy = accuracy_score(y, colorado_pred)
        print(f"Accuracy : {accuracy:.4f}")
        print(classification_report(y, colorado_pred))
        print(confusion_matrix(y, colorado_pred))
        results[model_name] = accuracy
        print(results)

    # Interpretabilite & Equite
    features = pd.read_csv('alt_acsincome_ca_features_85.csv')
    labels = pd.read_csv('alt_acsincome_ca_labels_85.csv')
    X, y = features, labels['PINCP'].ravel()
    #Correlations Initiales
    data = features.copy()
    data['PINCP'] = labels
    plot_correlation_matrix(data, 'PINCP')

    #Correlations RF
    data = features.copy()
    model = joblib.load("RandomForest_BestModel_0819.joblib")
    y_pred = model.predict(X)
    data['PREDICTION_RF'] = y_pred
    plot_correlation_matrix(data, 'PREDICTION_RF')

    #Correlations AB
    data = features.copy()
    model = joblib.load("AdaBoost_BestModel_0810.joblib")
    y_pred = model.predict(X)
    data['PREDICTION_AB'] = y_pred
    plot_correlation_matrix(data, 'PREDICTION_AB')

    #Correlations GB
    data = features.copy()
    model = joblib.load("GradientBoosting_BestModel_0827.joblib")
    y_pred = model.predict(X)
    data['PREDICTION_GB'] = y_pred
    plot_correlation_matrix(data, 'PREDICTION_GB')

    #Correlations Stacking
    data = features.copy()
    model = joblib.load("Stacking_BestModel_0822.joblib")
    y_pred = model.predict(X)
    data['PREDICTION_Stacking'] = y_pred
    plot_correlation_matrix(data, 'PREDICTION_Stacking')

    # RF ordre importance attributs
    plot_feature_importance(X, y, model_type='RandomForest')
    
    # Equite
    # Evaluer le biais du genre
    evaluate_gender_bias(X, y, gender_column='SEX')
    # Fairness
    model = joblib.load("RandomForest_BestModel_0819.joblib")
    fairness_metrics = evaluate_fairness_by_gender(model, X, y, gender_column='SEX')
    print("Métriques d'équité par genre :")
    for gender, metrics in fairness_metrics.items():
        print(f"\nGenre : {gender}")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name} : {metric_value:.4f}")




    
main()
