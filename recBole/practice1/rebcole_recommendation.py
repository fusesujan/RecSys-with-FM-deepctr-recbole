import pandas as pd
from recbole.quick_start import run_recbole
from recbole.config import Config


# Load user-item interaction data
interaction_df = pd.read_csv('u.data', sep='\t', names=[
                             'user_id', 'item_id', 'rating', 'timestamp'])

# Load movie information
movie_info_df = pd.read_csv(
    'u.item', sep='|', encoding='latin-1', names=['item_id', 'title'], usecols=[0, 1])

# Merge interaction and movie information
data_df = pd.merge(interaction_df, movie_info_df, on='item_id')

# Define RecBole dataset format
data_df = data_df[['user_id', 'item_id', 'rating', 'title']]

# Save the data as a CSV file for RecBole
data_df.to_csv('movielens.csv', index=False)


# Define the configuration for your recommendation system
config_file = 'config.yaml'

# Run the recommendation system experiment
run_recbole(config_file)
