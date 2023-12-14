import os
import gc
from .to_chunks import load_ratings, split_into_chunks

dir_name = 'Netflix'
chunk_dir = 'data/full'
probe_file = 'probe.txt'

training_dir = 'data/training'
probe_dir = 'data/probe'

def dump_to_file(filename, ratings):
    f = open(filename, 'w')
    for movie_id in ratings:
        users_ratings = ratings[movie_id]

        f.write(f"{movie_id}:\n")
        for user_id in users_ratings:
            f.write(f"{user_id},{users_ratings[user_id]}\n")
    
    f.close()


def files_in_dir(files_dir):
    return [f for f in os.listdir(files_dir) if os.path.isfile(os.path.join(files_dir, f))]

def files_in_dir_v1(files_dir):
    return [os.path.join(files_dir, f) for f in os.listdir(files_dir) if os.path.isfile(os.path.join(files_dir, f))]

if __name__ == '__main__':
    chunk_files = files_in_dir(chunk_dir)
    probe = load_ratings(os.path.join(dir_name, probe_file))

    for chunk in chunk_files:
        chunk_name = os.path.join(chunk_dir, chunk)
        data = load_ratings(chunk_name)

        for movie_id in data:
            chunk_ratings = probe.get(movie_id, None)
            if chunk_ratings == None:
                continue

            for probe_rating in chunk_ratings:
                probe[movie_id][probe_rating] = data[movie_id][probe_rating]
                del data[movie_id][probe_rating]

        dump_to_file(os.path.join(training_dir, chunk), data)

        del data
        gc.collect()

    split_into_chunks(probe, "probe", dirpath=probe_dir)