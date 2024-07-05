import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine

# Function to scrape player stats from Pro Football Reference
def scrape_player_stats(year):
    url = f'https://www.pro-football-reference.com/years/{year}/fantasy.htm'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table
    table = soup.find('table', {'id': 'fantasy'})
    df = pd.read_html(str(table))[0]
    
    # Clean the data
    df.columns = df.columns.droplevel(0)  # Drop multi-level columns
    df = df[df.Rk != 'Rk']  # Remove header rows that appear within the data
    df = df.fillna(0)  # Replace NaN values with 0
    df = df.reset_index(drop=True)
    return df

# Scrape data for multiple years
years = range(2010, 2024)
all_data = pd.DataFrame()
for year in years:
    year_data = scrape_player_stats(year)
    year_data['Year'] = year
    all_data = pd.concat([all_data, year_data])

# Save the data to a CSV file
all_data.to_csv('historical_player_stats.csv', index=False)

# Set up the SQLite database
engine = create_engine('sqlite:///nfl_stats.db')

# Load the data from the CSV file
data = pd.read_csv('historical_player_stats.csv')

# Store the data in the database
data.to_sql('player_stats', con=engine, if_exists='replace', index=False)

# Load the data into a DataFrame for preprocessing
data = pd.read_sql('player_stats', con=engine)

# Convert columns to appropriate data types
data['Year'] = data['Year'].astype(int)
data['FantasyPoints'] = data['FantPt'].astype(float)

# Handle missing values
data = data.fillna(0)

# Feature engineering
data['MovingAvgFantasyPoints'] = data.groupby('Player')['FantasyPoints'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
data['TeamPerformance'] = data.groupby(['Tm', 'Year'])['FantasyPoints'].transform('mean')

# Select relevant features
features = ['Player', 'Year', 'Tm', 'FantasyPoints', 'MovingAvgFantasyPoints', 'TeamPerformance']
data = data[features]

# Store the preprocessed data back into the database
data.to_sql('preprocessed_player_stats', con=engine, if_exists='replace', index=False)

print("Data scraping, storage, and preprocessing completed successfully.")
