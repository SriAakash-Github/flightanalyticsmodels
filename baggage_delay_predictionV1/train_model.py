import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# === Load dataset ===
file_path = r'C:\Users\sriba\OneDrive\Desktop\reg and class\testing\baggage_dataset.csv'
df = pd.read_csv(file_path)

# === Remove outliers in bag_weight_kg (above 80 kg) ===
df = df[df['bag_weight_kg'] <= 80]

# === Use baggage_delayed_minutes for regression ===
df['baggage_delayed_minutes'] = pd.to_numeric(df['baggage_delayed_minutes'], errors='coerce').fillna(0)

# === Cap extreme delays to 120 minutes ===
df['baggage_delayed_minutes'] = df['baggage_delayed_minutes'].clip(upper=120)

# === Create classification target: delayed if more than 15 mins ===
df['baggage_delayed'] = df['baggage_delayed_minutes'].apply(lambda x: 1 if x > 15 else 0)

# === Log-transform regression target to reduce skew ===
df['log_delay_minutes'] = np.log1p(df['baggage_delayed_minutes'])

# === Select input features ===
input_features = [
    'origin_airport', 'destination_airport',
    'departure_time', 'actual_departure_time',
    'arrival_time', 'check_in_time',
    'weather_condition', 'number_of_bags', 'bag_weight_kg',
    'day_of_week', 'is_international', 'extra_baggage',
    'travel_duration_min'
]
classification_target = 'baggage_delayed'
regression_target = 'log_delay_minutes'

df = df[input_features + [classification_target, regression_target]].copy()

# === Time feature engineering ===
for col in ['departure_time', 'actual_departure_time', 'arrival_time', 'check_in_time']:
    df[col] = pd.to_datetime(df[col].astype(str).str[-5:], format='%H:%M', errors='coerce')
    df[f'{col}_hour'] = df[col].dt.hour
    df[f'{col}_minute'] = df[col].dt.minute

# Create route feature
df['route'] = df['origin_airport'] + '-' + df['destination_airport']

# Treat day_of_week as string
df['day_of_week'] = df['day_of_week'].astype(str)

# === Final model features ===
final_features = [
    'origin_airport', 'destination_airport', 'route',
    'weather_condition', 'number_of_bags', 'bag_weight_kg',
    'day_of_week', 'is_international', 'extra_baggage',
    'departure_time_hour', 'departure_time_minute',
    'actual_departure_time_hour', 'actual_departure_time_minute',
    'arrival_time_hour', 'arrival_time_minute',
    'check_in_time_hour', 'check_in_time_minute',
    'travel_duration_min'
]

X = df[final_features]
y_class = df[classification_target]
y_reg = df[regression_target]

# === Preprocessing pipeline ===
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

numerical_pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
)

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# === Classification Pipeline (Ensemble) ===
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

ensemble_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('rf', rf),
        ('lr', lr)
    ],
    voting='soft'
)

clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('ensemble', ensemble_clf)
])

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, stratify=y_class, random_state=42
)
clf_pipeline.fit(X_train_c, y_train_c)
y_pred_c = clf_pipeline.predict(X_test_c)

print('\n==== Classification Report (Voting Ensemble) ====')
print(classification_report(y_test_c, y_pred_c))
print('Confusion Matrix:\n', confusion_matrix(y_test_c, y_pred_c))
print('Accuracy:', accuracy_score(y_test_c, y_pred_c))
print('F1 Score:', f1_score(y_test_c, y_pred_c))

# Save classification model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf_pipeline, f)

# === Regression Pipeline with Hyperparameter Tuning ===
reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [3, 4, 5, 6, 8],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__subsample': [0.6, 0.8, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0]
}

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg_search = RandomizedSearchCV(
    reg_pipeline,
    param_distributions=param_grid,
    n_iter=10,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

reg_search.fit(X_train_r, y_train_r)

best_reg_model = reg_search.best_estimator_

# Predict in log scale, then convert back
y_pred_log = best_reg_model.predict(X_test_r)
y_pred_minutes = np.expm1(y_pred_log)
y_test_minutes = np.expm1(y_test_r)

print('\n==== Regression Report (TUNED - Predicted in Minutes) ====')
print('Best Params:', reg_search.best_params_)
print('MAE:', mean_absolute_error(y_test_minutes, y_pred_minutes))
print('RMSE:', np.sqrt(mean_squared_error(y_test_minutes, y_pred_minutes)))
print('R² Score:', r2_score(y_test_minutes, y_pred_minutes))

# Save regression model
with open('model_reg.pkl', 'wb') as f:
    pickle.dump(best_reg_model, f)

print('\n✅ Both models trained and saved successfully.')
