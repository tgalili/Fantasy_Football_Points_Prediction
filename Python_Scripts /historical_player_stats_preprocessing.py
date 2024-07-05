import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Define the file path for historical player stats
player_stats_file = 'historical_player_stats.csv'

# Check if the file exists
if not os.path.exists(player_stats_file):
    raise FileNotFoundError(f"{player_stats_file} not found. Please ensure the file is in the correct directory.")

# Load data
player_stats = pd.read_csv(player_stats_file)

# Cleaning data
player_stats.fillna(0, inplace=True)
player_stats.drop_duplicates(inplace=True)

# Creating features: Player Statistics
# Example features: Weekly averages, rolling averages, performance ratios
player_stats['weekly_avg_points'] = player_stats.groupby('player_id')['points'].transform('mean')
player_stats['rolling_avg_points_3w'] = player_stats.groupby('player_id')['points'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
player_stats['td_to_int_ratio'] = player_stats['touchdowns'] / (player_stats['interceptions'] + 1)

# Additional features (if available in your dataset)
if 'team_points' in player_stats.columns:
    player_stats['avg_team_points'] = player_stats.groupby('team_id')['team_points'].transform('mean')
if 'wins' in player_stats.columns and 'losses' in player_stats.columns:
    player_stats['win_loss_ratio'] = player_stats['wins'] / (player_stats['losses'] + 1)

# Selecting relevant features
features_to_scale = ['weekly_avg_points', 'rolling_avg_points_3w', 'td_to_int_ratio']
if 'avg_team_points' in player_stats.columns:
    features_to_scale.append('avg_team_points')
if 'win_loss_ratio' in player_stats.columns:
    features_to_scale.append('win_loss_ratio')

# Scaling features
scaler = StandardScaler()
player_stats[features_to_scale] = scaler.fit_transform(player_stats[features_to_scale])

# Define the final feature set and target variable
features = player_stats[features_to_scale]
target = player_stats['points']  # Weekly points as the target variable

# Displaying the first few rows of the final feature set for verification
print(features.head())
print(target.head())
