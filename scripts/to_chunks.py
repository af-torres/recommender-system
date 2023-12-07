import os
import gc

dir_name = 'Netflix'
training_files = ['combined_data_1.txt', 'combined_data_2.txt', 'combined_data_3.txt', 'combined_data_4.txt']

def load_ratings(file_name) -> dict:    
    ratings = {}

    f = open(file_name, 'r')
    data = f.readlines()
    f.close()
    
    movie_id = None
    user_id = None
    
    for line in data:
        if ':' in line:
            movie_id = line[:-2]
            ratings[movie_id] = {}
            continue
        
        user_data = line.split(',')
        user_id = user_data[0].replace('\n', '')
        if len(user_data) == 1:
            ratings[movie_id][user_id] = 0
        else:
            ratings[movie_id][user_id] = ','.join([user_data[1], user_data[2].replace('\n','')])

    return ratings

def split_name(name, idx, dirpath):
    return f"{dirpath}/{name}_{idx}.txt" 

def split_into_chunks(ratings, name, chunks = 20, dirpath = "data/full"):
    split_rate = int(len(ratings) / chunks)
    split_index = 0

    f = open(split_name(name, split_index, dirpath), 'w')
    for idx, movie_id in enumerate(ratings):
        users_ratings = ratings[movie_id]

        f.write(f"{movie_id}:\n")
        for user_id in users_ratings:
            f.write(f"{user_id},{users_ratings[user_id]}\n")

        if (idx + 1) % split_rate == 0:
            print(f"closing file {idx, split_index}")
            f.close()

            split_index = int((idx + 1) / split_rate)
            f = open(split_name(name, split_index, dirpath), "w")

    f.close()

if __name__ == '__main__':    
    for idx, fn in enumerate(training_files):
        full_name = os.path.join(dir_name, fn)
        ratings = load_ratings(full_name)
        
        split_into_chunks(ratings, idx, 40)

        del ratings
        gc.collect()
