import psycopg
import pandas as pd

from .clean_training import files_in_dir_v1
from .to_csv import csv_training_dir
from .shift_ids import users_shift_key, ratings_key, movies_shift_key

connection_str = (
    "postgresql://postgres:mysecretpassword@localhost:5432/postgres"
)

def build_sql_insert(df: pd.DataFrame) -> str:
    insert_statement = (
        "INSERT INTO ratings \
        (user_id, movie_id, rating)\n\
        VALUES\n"
    )

    for index, row in df.iterrows():
        sep = ","
        if index == len(df) -1:
            sep = ";"
        
        insert_statement += f"({row[users_shift_key]},{row[movies_shift_key]},{row[ratings_key]}){sep}"
    
    return insert_statement

if __name__ == '__main__':
    training_files = files_in_dir_v1(csv_training_dir)

    with psycopg.connect(connection_str) as conn:
        with conn.cursor() as cur:
            for file_name in training_files:
                df = pd.read_csv(file_name)
                statement = build_sql_insert(df)
                
                cur.execute(statement)
                
            conn.commit()
