from typing import List
import os
from to_chunks import load_ratings
from clean_training import files_in_dir, training_dir, probe_dir

csv_training_dir = 'data/csv/training'
csv_probe_dir = 'data/csv/probe'

def to_csv(ratings, filename):
    output = 'movie_id,user_id,rating,date\n'

    for movie_id in ratings:
        for user_id in ratings[movie_id]:
            rating = ratings[movie_id][user_id]
            output += f"{movie_id},{user_id},{rating}\n"

    f = open(filename, 'w')
    f.write(output)
    f.close()

if __name__ == '__main__':
    training_files: List[str] = files_in_dir(training_dir)
    probe_files: List[str] = files_in_dir(probe_dir)

    for file_name in training_files:
        ratings = load_ratings(os.path.join(training_dir, file_name))
        to_csv(ratings, os.path.join(csv_training_dir, file_name.replace('txt', 'csv')))

    for file_name in probe_files:
        ratings = load_ratings(os.path.join(probe_dir, file_name))
        to_csv(ratings, os.path.join(csv_probe_dir, file_name.replace('txt', 'csv')))
