import logging
import numpy as np
import pandas as pd

from typing import Tuple, Literal
from scripts.shift_ids import users_shift_key, movies_id_key


iter_key = Literal['movie', 'user']


log = logging.getLogger(__name__)


class Generator:
    def __init__(
            self, id_key,
            unique_groups: np.ndarray,
            connection,
            files=None,
            offset=200
        ):
        log.debug('init generator for %s', id_key)
        log.debug('generator range from %s to %s', unique_groups[0], unique_groups[-1])

        self.__id_key = id_key
        self.__unique_groups = unique_groups
        self.__files = files
        self.connection = connection

        self.__next = 0 # index of next load
        self.__offset = offset

        self.__max = len(self.__unique_groups)
        
        self.done = False

    def __load_from_files(self, index: np.ndarray):
        result = pd.DataFrame()

        for file_name in self.__files:
            df = pd.read_csv(file_name, index_col=self.__id_key)
            mask = np.isin(df.index.to_numpy(), index)

            chunk = df[mask]
            result = pd.concat([result, chunk])

        return result
    
    def __load_from_database(self, index: np.ndarray):
        columns = ["user_id", "movie_id", "rating"]
        query_statement = (
            f"SELECT {','.join(columns)} "
            "FROM ratings "
            f"WHERE {self.__id_key} in "
            f"({','.join(['%s' for _ in range(len(index))])})"
        )

        cur = self.connection.cursor()
        result = None
        try:
            cur.execute(query_statement, index.tolist())
        except Exception as e:
            log.error(e, exc_info=True)
            self.connection.rollback()
        else:
            raw_data = cur.fetchall()
            result = pd.DataFrame(np.array(raw_data), columns=columns)
            result = result.set_index(self.__id_key)
            self.connection.commit() # clean up connection to perform future queries
        
        return result
    
    def __iter__(self):
        self.done = False
        return self

    def __next__(self) -> Tuple[np.ndarray, pd.DataFrame]:
        if self.done:
            raise StopIteration

        done = False
        start = self.__next
        stop = self.__next + self.__offset + 1
        
        if stop >= self.__max:
            stop = self.__max
            done = True

        index = self.__unique_groups[start:stop]

        data = self.__load_from_database(index)

        self.__next += self.__offset
        self.done = done

        return (index, data)


def make(
        ids_df: pd.DataFrame,
        group_key: iter_key,
        connection,
    ) -> Generator:
    id_key = None
    if group_key == 'user':
        id_key = 'user_id'
        shift_key = users_shift_key
        offset = 3500
    else:
        id_key = 'movie_id'
        shift_key = movies_id_key
        offset = 200
    
    unique_groups = ids_df[shift_key].to_numpy()

    return Generator(id_key, unique_groups, connection, offset=offset)
