import pandas as pd

from .clean_training import files_in_dir_v1
from .to_csv import csv_training_dir, csv_probe_dir

users_csv = 'data/csv/training_unique_users.csv'
movies_csv = 'data/csv/training_unique_movies.csv'

users_id_key = 'user_id'
users_shift_key = 'shifted_user'

movies_id_key = 'movie_id'
movies_shift_key = 'movie_id'

ratings_key = 'rating'

def load_user_id_to_shift_id():
    return pd.read_csv(users_csv, index_col=users_id_key).iloc[:, 0]

if __name__ == '__main__':
    users_shift = load_user_id_to_shift_id()

    all_files = []
    all_files.extend(files_in_dir_v1(csv_training_dir))
    all_files.extend(files_in_dir_v1(csv_probe_dir))

    for file_name in all_files:
        df = pd.read_csv(file_name)
        df[users_shift_key] = users_shift[df[users_id_key]].values
        df.to_csv(file_name, index=False)
