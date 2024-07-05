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
