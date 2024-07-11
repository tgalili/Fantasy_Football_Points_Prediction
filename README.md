# Fantasy Football Points Prediction
Predicting fantasy football points for the 2024 NFL season using Python and SQL

## Contents

- [Introduction](#introduction)
- [Data Preparation and Description Using SQLite](#data-preparation-and-description-using-sqlite)
- [Reading and Visualizing the Dataset on Python](#reading-and-visualizing-the-dataset-on-python)
- [Feature Engineering](#feature-engineering)
- [Train and Test Split with Cross Validation](#train-and-test-split-with-cross-validation)
- [Regression Model Implementations](#regression-model-implementations)
  - [Scaling and Polynomial Features](Scaling-and-Polynomial-Features)
  - [Lasso](#b-lasso)
  - [Ridge](#c-ridge)
  - [Gradient Boosting XR](#d-gradient-boosting-xr)
- [Validation](#validation-of-best-model)
- [Conclusion](#conclusion)
- [Citations](#citations)

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
## Reading and Visualizing the Dataset in Python
Once the necessary data contained in a SQLite database was obtained, the data was read into Python as a DataFrame. This step is crucial for ensuring that the data is well-structured and easily accessible for further analysis and modeling. To ensure the replicability of results and consistency in future comparisons, I maintained a consistent random state during sampling.

1. Reading the Data:
   The data was read from the SQLite database nfl_stats.db using the SQLAlchemy library. This allowed me to seamlessly load the player statistics into a pandas DataFrame, which provides powerful data manipulation capabilities.

2. Summary Statistics:
   To familiarize myself with the dataset, I generated summary statistics. Using df.info(), I visualized the datatypes and the number of non-null entries for each variable. This dataset contained various data types, including integers and floats, and provided an overview of the data's completeness.
   Next, df.describe() offered a descriptive statistics summary for each variable, displaying the count, mean, standard deviation, minimum, maximum, and percentiles. This step helped in understanding the central tendencies and variability within the dataset.

3. Handling Missing Values:
   Missing values were identified and addressed using forward fill and backward fill methods. Forward fill propagates the last valid observation forward to fill gaps, while backward fill does the same but in reverse. This ensured that the dataset was complete and ready for analysis without introducing biases from missing data.

4. Visualizing the Data:
   Visualization is a critical step in data analysis as it helps in uncovering patterns and relationships that are not immediately apparent from raw data. Using Python’s visualization libraries, such as Matplotlib and Seaborn, I created several plots to explore the data.
      * Distribution of Weekly Points: A histogram was used to visualize the distribution of the target variable FantPt (weekly points). This plot revealed how fantasy points are distributed across players and highlighted any skewness or outliers in the data.
      * Distribution of Total Points: Similarly, a histogram was plotted for the total points (PPR). This helped in understanding the overall performance distribution of players across the season.
      * Scatter Plot of Player Age vs Total Points: To examine the relationship between player age and total points, a scatter plot was created. This visualization provided insights into whether age had any significant effect on a player's total fantasy points.
      * Correlation Matrix: A correlation matrix heatmap was generated to visualize the relationships between various features and the target variable. Correlations are represented as values between +1 and -1, where +1 indicates a perfect positive correlation, -1 indicates a perfect negative correlation, and 0 indicates no correlation. The heatmap highlighted which features were strongly correlated with FantPt and with each other, informing feature engineering decisions.


| ![Original Variable Scatter Plots](https://github.com/tgalili/Fantasy_Football_Points_Prediction/blob/main/images/Linear%20Regression.png) | 
|:--:| 
| *Figure 1 - Scatter Plot* |

| ![Original Variable Correlation Heat Map](https://github.com/tgalili/Fantasy_Football_Points_Prediction/blob/main/images/Correlation%20Matrix.png) |
|:--:| 
| *Figure 2 - Correlation Heat Map* |

These visualizations and summaries were instrumental in identifying key patterns and relationships within the data. For instance, the correlation matrix revealed significant positive correlations between team performance and weekly fantasy points, indicating that team dynamics play a crucial role in individual player performance. Understanding these correlations is essential for feature engineering and model building, ensuring that the most relevant and predictive features are used in the models.

By thoroughly reading and visualizing the dataset in Python, I ensured a comprehensive understanding of the data’s structure and relationships. This groundwork is essential for effective feature extraction, model training, and evaluation, ultimately leading to more accurate predictions of fantasy football points.
## Feature Engineering
In this project, feature engineering was used to create new variables that enhance the prediction of fantasy football points:
1. Rolling Average of Fantasy Points:
   * 3-game Rolling Average: Smoothing recent performance with a 3-game rolling average.
2. Exceptional Performance Indicator:
   * Binary Indicator: Flagging games where players scored over 20 points.
3. Team Performance Metrics:
   * Team Average Fantasy Points: Average points per game for each team.
   * Normalized Team Performance: Team performance relative to the league average.
4. Feature and Target Selection:
   * Key features: FantPt_rolling_avg, exceptional_performance, normalized_team_performance, Age.
   * Target variable: FantPt.
5. Handling Missing and Infinite Values:
   * Replaced infinite values and dropped missing values.
6. Feature Scaling:
   * Standardized features using StandardScaler.
7. Data Splitting:
   * Split data into training and testing sets.

## Regression Model Implementations
To predict fantasy football points, I employed multiple regression models, focusing on feature scaling, combinations of variables, and polynomial features to find the best model with the lowest Mean Squared Error (MSE).

#1. Scaling and Polynomial Features:

  * Standardized the variables to avoid multicollinearity, especially for polynomial and interaction terms.
  * Used sklearn's PolynomialFeatures to create interaction terms and polynomial features up to the 2nd degree.

2. Feature Combinations:

  * Implemented a function to try all possible combinations of feature variables.
  * Tested combinations to find the best-performing model with the lowest MSE.

A. Ordinary Least Squares (OLS)
  * Linear Regression: Fits a linear model with coefficients to minimize the residual sum of squares between observed and predicted targets.
  * Best Model: Identified using "neg_mean_squared_error" as the scoring metric. The best combination included features such as 'FantPt_rolling_avg', 'exceptional_performance', and 'normalized_team_performance', resulting in the lowest MSE.
B. Random Forest Regressor
  * Random Forest: An ensemble method that fits multiple decision trees and averages their predictions to improve accuracy and control overfitting.
  * Best Model: The combination of features provided robust performance, with significant improvements over linear models in terms of accuracy and MSE.
C. Gradient Boosting
  * XGBoost: An advanced ensemble method that builds successive trees to correct errors of previous ones, enhancing model accuracy.
  * Hyperparameter Tuning: Used RandomizedSearchCV for hyperparameter tuning, optimizing parameters like max_depth, learning_rate, and subsample ratio.
  * Best Model: Achieved an MSE of 2279.02 and an R² of 86.38%, indicating a highly accurate model suitable for large datasets.
By systematically implementing and comparing these regression models, I ensured a robust approach to predicting fantasy football points, leveraging both linear and ensemble methods to achieve high accuracy and low error rates.
