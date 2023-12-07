import logging
import os
import time
from typing import Tuple
import numpy as np
import pandas as pd
import psycopg
import generator

from scripts.shift_ids import users_csv, movies_csv, movies_shift_key
from scripts.to_db import connection_str


rating_key = 'rating'
users_shift_key = 'user_id'


logging.basicConfig(level=os.getenv('LOG_LEVEL') or logging.INFO)
log = logging.getLogger(__name__)


def load_init_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv(users_csv)
    movies = pd.read_csv(movies_csv)

    return (users, movies)


def init_weights(
        users: pd.DataFrame,
        movies: pd.DataFrame,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    total_users = users.shape[0]
    total_movies = movies.shape[0]

    u = np.random.normal(0, 1, size=(total_users + 1, k))
    v = np.random.normal(0, 1, size=(total_movies + 1, k))

    log.info(f"total movies: {total_movies}")
    log.info(f"total users: {total_users}")

    return (u, v)


class lm:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        # least square estimates
        betas: np.ndarray = np.dot(
            np.linalg.inv(np.dot(x.T, x)),
            np.dot(
                x.T,
                y
            ),
        )

        residual: np.ndarray = np.dot(x, betas) - y
        rdf: int = x.shape[0] - betas.shape[0]

        self.betas = betas
        self.residual = residual
        self.rdf = rdf


def train(
        users: pd.DataFrame,
        movies: pd.DataFrame,
        uGenerator: generator.Generator,
        vGenerator:generator.Generator,
        k = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
    u, v = init_weights(users, movies, k)

    for movies, batch in vGenerator:
        log.debug("movies batch from id %s to %s", movies[0], movies[-1])
        log.debug("total batch size %s", batch.shape)
        for m_id in movies:
            users: np.ndarray = batch.loc[[m_id], users_shift_key].to_numpy()
            if len(users) < k:
                continue # if users < k, the matrix is invertible

            x = u[users, :]
            y: np.ndarray = batch.loc[[m_id], rating_key].to_numpy()

            model = lm(x, y)

            v[m_id, :] = model.betas

    for users, batch in uGenerator:
        log.debug("users batch from id %s to %s", users[0], users[-1])
        log.debug("total batch size %s", batch.shape)
        for u_id in users:
            movies = batch.loc[[u_id], movies_shift_key].to_numpy()
            if len(movies) < k: # if movies < k, the matrix is invertible
                continue

            x = v[movies, :]
            y = batch.loc[[u_id], rating_key].to_numpy()

            model = lm(x, y)

            u[u_id, :] = model.betas

    return (u, v)


if __name__ == '__main__':
    log.debug('program start')
    users, movies = load_init_data()

    try:
        conn = psycopg.connect(connection_str)
        uGenerator = generator.make(users, 'user', conn)
        vGenerator = generator.make(movies, 'movie', conn)

        log.info("start training")
        start = time.perf_counter()
        u, v = train(users, movies, uGenerator, vGenerator)
        log.info("Total training time of one full ALS step: %s", time.perf_counter() - start)
        log.info("training completed. saving weights")
        
        np.save("weights_u.npy", u)
        np.save("weights_v.npy", v)
    except Exception as e:
        log.error(e, exc_info=True)
    finally:
        log.info("closing database connection")
        conn.close()
