# Fantasy Football Points Prediction
Predicting fantasy football points for the 2024 NFL season using Python and SQL

## Introduction
The goal of this project is to forecast fantasy football points for the 2024 NFL season by investigating and utilizing different machine learning regression algorithms. This project aims to develop models that forecast weekly and total fantasy points using historical player statistics and team dynamics, in a manner akin to the method used to predict residential property prices based on physical and locational attributes.
To deal with missing values and guarantee quality, the data is first preprocessed. After that, an analysis is conducted to highlight the key attributes of the variables, including any correlations and discernible patterns. Important components designed to improve the predictive capacity of the models include normalized team performance metrics and rolling averages of fantasy points.
These features are fed into different machine learning regression algorithms, with the data being split into training and testing sets. Models, including Linear Regression and Random Forest Regressor, are trained and evaluated. The performance of these models is assessed using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²). The best model is identified based on its accuracy and performance on the validation set.
## Data Preparation and Description Using SQLite
The data preparation begins with loading the historical player statistics from a SQLite database. This process ensures that the data is well-structured and easily accessible for analysis and model training.
1. Connecting to SQLite Database: Using SQLAlchemy, the player statistics are retrieved from the nfl_stats.db database and loaded into a pandas DataFrame, facilitating easy data manipulation.
2. Initial Data Exploration: Basic exploration involves displaying the first few rows, summarizing data types and non-null counts, and generating descriptive statistics to understand the data structure and identify any issues.
3. Handling Missing Values: Missing data is addressed using forward fill and backward fill methods, which fill gaps by propagating the last valid observation forward or backward.
4. Feature Engineering: New features are created to enhance predictive power, including:
    * 3-game rolling average of fantasy points: Smooths recent player performance.
    * Exceptional performance indicator: Binary feature for high-scoring games.
    * Normalized team performance: Averages fantasy points per game at the team level, normalized against the league average.
