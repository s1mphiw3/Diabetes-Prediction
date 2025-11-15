# Updated Predicting Diabetes Notebook as Python Script

# Importing Libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
import os

# Data Loading with error handling
data_path = "UNESWADataSet.xlsx"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file {data_path} not found. Please check the path.")
df = pd.read_excel(data_path)

# Data Pre-Processing
# Dealing with Missing values, duplicates and datatypes
df['ScreenedFor'] = df['ScreenedFor'].replace('Hypertension_Diabetes_Diabetes','Hypertension_Diabetes')
df['ScreenedFor'] = df['ScreenedFor'].replace('_Diabetes_Diabetes','_Diabetes')
df['ScreenedFor'] = df['ScreenedFor'].replace('_Diabetes','Diabetes')

# Dropping unnecessary time related fields
df = df.drop(columns=['DiagnosisDate','TreatmentDate','HypertensionDiagnosisDate', 'HIVDiagnosisDate', 'ARTStartDate','TBStartDate'])

# Replacing null values
fillna_dict = {
    'Impotence': 1,
    'CVAStroke': 1,
    'PatientLifestyle': 'None',
    'FamilyHistory': 'None',
    'Smoking': 'Not Smoking',
    'Alcohol': 'No Alcohol',
    'NutritionalStatus': 'Normal',
    'RiskOutcome': 'Not at risk',
    'FBGOutcome': 'Normal',
    'RBGOutcome': 'Normal',
    'ScreeningResult': 'Normal',
    'Complication': 'None',
    'TestResult': 'Normal',
    'Tested': 'No'
}
for col, val in fillna_dict.items():
    df[col] = df[col].fillna(val)

median_fill_cols = ['PeakFlowmeter', 'BPDiastolic', 'BPSystolic', 'BMI', 'Score', 'GlucoseFasting', 'GlucoseRandom', 'HbA1C']
for col in median_fill_cols:
    df[col] = df[col].fillna(df[col].median())

# Convert datatypes
df['Impotence'] = df['Impotence'].astype(int)
df['CVAStroke'] = df['CVAStroke'].astype(int)

# Drop datetime entries in AgeGroup
from datetime import datetime
df = df[~df['AgeGroup'].apply(lambda x: isinstance(x, datetime))]

# Encoding categorical fields with LabelEncoder
categorical_cols = ['AgeGroup','PatientLifestyle','Sex', 'FamilyHistory', 'Smoking','Alcohol','NutritionalStatus','RiskOutcome', 
                    'FBGOutcome', 'RBGOutcome', 'ScreeningResult','Screened', 'Complication', 'TestResult','Death','ScreenedFor','Tested', 'DiabetesControl']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Drop columns that leak future information or are identifiers
drop_cols = ['VisitID','ClientID','FacilityCode','AgeGroupBins','VisitDate','DiabetesControl','Age','TreatedforHypertension','TreatedForDiabetes','Death',
             'FBGOutcome', 'RBGOutcome', 'ScreeningResult', 'TestResult', 'Complication','DiagnosedWithHypertension', 'GlucoseFasting', 'GlucoseRandom', 'Tested', 'HbA1C']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Feature selection using SelectKBest
X = df.drop(columns=['DiagnosedWithDiabetes'])
y = df['DiagnosedWithDiabetes']

from sklearn.feature_selection import SelectKBest, chi2
best_features = SelectKBest(score_func=chi2, k=15)
fit = best_features.fit(X, y)
X = X.loc[:, best_features.get_feature_names_out()]

# Balancing target variable using SMOTE
ros = SMOTE(random_state=15)
X_res, y_res = ros.fit_resample(X, y)
df_balanced = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.DataFrame(y_res, columns=['DiagnosedWithDiabetes'])], axis=1)

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_balanced.drop(columns=['DiagnosedWithDiabetes']))
y_balanced = df_balanced['DiagnosedWithDiabetes']

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_balanced, stratify=y_balanced, test_size=0.3, random_state=15)

# Model definitions
models = {
    'RandomForest': RandomForestClassifier(max_depth=5, min_samples_split=10, class_weight='balanced', n_estimators=200),
    'MLP': MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000),
    'GradientBoosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Hyperparameter tuning for RandomForest (example)
param_grid_rf = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [5, 10, 15],
    'n_estimators': [100, 200]
}
grid_rf = GridSearchCV(models['RandomForest'], param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

# Train and evaluate models
for name, model in models.items():
    if name == 'RandomForest':
        model = best_rf
    else:
        model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"{name} Accuracy:", accuracy_score(y_test, predictions))
    print(f"{name} Balanced Accuracy:", balanced_accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

# Save best model
pickle.dump(best_rf, open('model.pkl', 'wb'))

# SHAP explainability
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test[0,:])
