import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle
import os

# Remove old model files if they exist
for model_file in ['linear_model.pkl', 'rf_model.pkl', 'scaler.pkl']:
    if os.path.exists(model_file):
        os.remove(model_file)

# Load the Data
engine = create_engine('sqlite:///nfl_stats.db')
df = pd.read_sql('SELECT * FROM player_stats', con=engine)

# Basic Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Handling Missing Values
print(df.isnull().sum())
df = df.fillna(method='ffill')
df = df.fillna(method='bfill')

# Exploratory Data Analysis (EDA) with Visualizations
sns.set_style("whitegrid")

# Distribution of weekly points (using FantPt as the weekly points metric)
plt.figure(figsize=(10, 6))
sns.histplot(df['FantPt'], bins=30, kde=True)
plt.title('Distribution of Weekly Points')
plt.xlabel('Weekly Points (FantPt)')
plt.ylabel('Frequency')
plt.show()

# Distribution of total points (using PPR as an example of total points)
plt.figure(figsize=(10, 6))
sns.histplot(df['PPR'], bins=30, kde=True)
plt.title('Distribution of Total Points (PPR)')
plt.xlabel('Total Points (PPR)')
plt.ylabel('Frequency')
plt.show()

# Relationship between player age and total points (PPR)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='PPR', data=df)
plt.title('Player Age vs Total Points (PPR)')
plt.xlabel('Age')
plt.ylabel('Total Points (PPR)')
plt.show()

# Relationship between team (Tm) performance and weekly points (FantPt)
team_performance = df.groupby('Tm')['FantPt'].sum().reset_index()
team_performance.columns = ['Tm', 'team_performance']
df = df.merge(team_performance, on='Tm')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='team_performance', y='FantPt', data=df)
plt.title('Team Performance vs Weekly Points (FantPt)')
plt.xlabel('Team Performance (Total FantPt)')
plt.ylabel('Weekly Points (FantPt)')
plt.show()

# Correlation Analysis
correlation_matrix = df[['Age', 'FantPt', 'PPR', 'DKPt', 'FDPt', 'team_performance']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Exploring Categorical Variables
plt.figure(figsize=(12, 8))
sns.boxplot(x='FantPos', y='FantPt', data=df)
plt.title('Weekly Points (FantPt) by Player Position')
plt.xlabel('Position')
plt.ylabel('Weekly Points (FantPt)')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='Tm', y='PPR', data=df)
plt.title('Total Points (PPR) by Team')
plt.xlabel('Team')
plt.ylabel('Total Points (PPR)')
plt.xticks(rotation=90)
plt.show()

# Feature Engineering
df['FantPt_rolling_avg'] = df.groupby('Rk')['FantPt'].rolling(window=3).mean().reset_index(0, drop=True)
df['exceptional_performance'] = df['FantPt'].apply(lambda x: 1 if x > 20 else 0)

team_avg_fantpt = df.groupby('Tm')['FantPt'].mean().reset_index()
team_avg_fantpt.columns = ['Tm', 'team_avg_fantpt']
df = pd.merge(df, team_avg_fantpt, on='Tm')

league_avg_fantpt = df['team_avg_fantpt'].mean()
df['normalized_team_performance'] = df['team_avg_fantpt'] / league_avg_fantpt

features = ['FantPt_rolling_avg', 'exceptional_performance', 'normalized_team_performance', 'Age']
target = 'FantPt'
X = df[features]
y = df[target]

X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y[X.index]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f'Linear Regression - MAE: {mae_linear}, MSE: {mse_linear}, R2: {r2_linear}')

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest Regressor - MAE: {mae_rf}, MSE: {mse_rf}, R2: {r2_rf}')

# Compare Model Performance
print('Model Performance:')
print(f'Linear Regression - MAE: {mae_linear}, MSE: {mse_linear}, R2: {r2_linear}')
print(f'Random Forest Regressor - MAE: {mae_rf}, MSE: {mse_rf}, R2: {r2_rf}')

# Model Evaluation
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel('Actual Weekly Points (FantPt)')
plt.ylabel('Predicted Weekly Points (FantPt)')
plt.title('Linear Regression: Actual vs Predicted Weekly Points')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel('Actual Weekly Points (FantPt)')
plt.ylabel('Predicted Weekly Points (FantPt)')
plt.title('Random Forest Regressor: Actual vs Predicted Weekly Points')
plt.show()

# Visualize Residuals Distribution
plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred_linear, bins=30, kde=True)
plt.title('Linear Regression: Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred_rf, bins=30, kde=True)
plt.title('Random Forest Regressor: Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Feature Importance for Random Forest
importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Save the Linear Regression model
with open('linear_model.pkl', 'wb') as f:
    pickle.dump(linear_model, f)

# Save the Random Forest model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
