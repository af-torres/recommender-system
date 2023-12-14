from math import sqrt
from typing import Tuple
import numpy as np
import pandas as pd

from scripts.clean_training import files_in_dir_v1

def load_weights() -> Tuple[np.ndarray, np.ndarray]:
    u = np.load('weights_u.npy')
    v = np.load('weights_v.npy')

    return u, v


if __name__ == '__main__':
    u, v = load_weights()
    
    probe_files = files_in_dir_v1('data/csv/probe')
    batch_size = 200
    n = 0
    sse = 0
    for file_name in probe_files:
        df = pd.read_csv(file_name)
        
        for batch in range(1, int(len(df)/batch_size)):
            batch_df = df.loc[batch_size * (batch -1): batch_size * batch]
            users = batch_df['shifted_user'].to_numpy()
            movies = batch_df['movie_id'].to_numpy()
            ratings = batch_df['rating'].to_numpy()

            y_hat = np.sum(u[users, :] * v[movies, :], axis=1)

            e = y_hat - ratings
            sse += np.sum(np.dot(e.T, e))
            n += batch_df.shape[0]

    print(f'sse: {sse}')
    print(f'size: {n}')
    print(f'mse: {sse/n}')
    print(f'rmse: {sqrt(sse/n)}')
