from typing import List
import numpy as np
import pandas as pd

from scripts.clean_training import files_in_dir
from scripts.to_csv import csv_training_dir


if __name__ == '__main__':
    training_files: List[str] = files_in_dir(csv_training_dir)
    unique_movies = []
    unique_users = []
    for file_name in training_files:
        df = pd.read_csv(f'{csv_training_dir}/{file_name}')
        unique_movies.extend(df['movie_id'].unique())
        unique_users.extend(df['user_id'].unique())

    unique_users = np.unique(unique_users)
    unique_movies = np.unique(unique_movies)
    users_df = pd.DataFrame({'user_id': unique_users})
    movies_df = pd.DataFrame({'movie_id': unique_movies})

    users_df.to_csv('data/csv/training_unique_users.csv', index = False)
    movies_df.to_csv('data/csv/training_unique_movies.csv', index = False)
