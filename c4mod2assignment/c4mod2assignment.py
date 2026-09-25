import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

def print_with_tms(message):
    mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{mytimestamp}|{message}")

def people_wiki_dtype_dict():
    dtype_dict = {'URI':str, 'name':str, 'text':str}
    return dtype_dict

def read_data(path=".\\", file_name="data.csv", dtype_dict=None):
    mydata = pd.read_csv(
        os.path.join(path, file_name),
        sep=",",
        quotechar='"',
        dtype=dtype_dict
    )
    return mydata

def assess_dataframe(df):
    print_with_tms(f"rowcount {df.shape[0]} colcount {df.shape[1]}")
    print_with_tms(f"dtypes {dict(df.dtypes)}")
    print_with_tms(f"columns\n{df.columns.tolist()}")
    if PRINTSTUFF == True:
        print_with_tms("First 5 rows:")
        print_with_tms(df.head())

def read_word_map(path, file_name="word_map.json"):
    with open(
        os.path.join(path, file_name),
        mode="r",
        encoding="utf-8"
    ) as file:
        return json.load(file)

def read_sparse_npz(path, file_name="some_sparse_matrix.npz"):
    file_path = os.path.join(path, file_name)

    with np.load(file_path, allow_pickle=False) as archive:
        return sparse.csr_matrix(
            (
                archive["data"],
                archive["indices"],
                archive["indptr"]
            ),
            shape=tuple(archive["shape"])
        )

def assess_sparse_matrix(word_map, sparse_matrix):
    word_map_items = list(word_map.items())
    print(f"number of words in word_map {len(word_map_items)} {word_map_items[:2]} ... {word_map_items[-2:]}")
    print_with_tms(f"type(word_map) {type(word_map)}")
    print_with_tms(f"type(sparse_matrix) {type(sparse_matrix)}")
    print_with_tms(f"sparse_matrix.shape {sparse_matrix.shape}")
    print_with_tms(f"sparse_matrix.nnz {sparse_matrix.nnz}")
    print_with_tms(f"sparse_matrix.format {sparse_matrix.format}")
    print_with_tms(f"sparse_matrix.dtype {sparse_matrix.dtype}")
    index_to_word = {index: word for word, index in word_map.items()}
    first_row = sparse_matrix.getrow(0)
    if PRINTSTUFF == True:
        for word_index, count in zip(first_row.indices, first_row.data):
            print(index_to_word[word_index], count)

def find_index_by_name(df, name):
    i = df.index[df["name"] == name][0]
    return i

def find_and_print_neighbours(df, model, word_count, i, n_neighbors):
    distances, indices = model.kneighbors(word_count[i], n_neighbors=n_neighbors)
    '''
    distances.shape == (1, 10)
    indices.shape   == (1, 10)
    '''

    neighbors = pd.DataFrame({
        "id": indices[0],
        "distance": distances[0]
    })

    result = (
        df.reset_index()
        .rename(columns={"index": "id"})
        .loc[neighbors["id"]]
        .reset_index(drop=True)
        .merge(neighbors, on="id")
        .sort_values("distance")
    )

    print(result[["id", "name", "distance"]])


def top_words(df, name, word_count, word_map):
    i = find_index_by_name(df, name)
    row = word_count.getrow(i)

    index_to_word = {
        index: word
        for word, index in word_map.items()
    }

    result = pd.DataFrame({
        "word": [index_to_word[index] for index in row.indices],
        "count": row.data
    })

    return result.sort_values(
        "count",
        ascending=False
    ).reset_index(drop=True)


PRINTSTUFF = False

print_with_tms("script started")

path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course04", "module02", "data")

'''
The input file is comma separated and with text qualifier double quotes
'''
file_name_inp = "people_wiki.csv"
file_name_word_map = "people_wiki_map_index_to_word.json"
file_name_word_count = "people_wiki_word_count.npz"
file_name_tf_idf = "people_wiki_tf_idf.npz"

all_data_df = read_data(path=path, file_name=file_name_inp, dtype_dict=people_wiki_dtype_dict())
assess_dataframe(df=all_data_df)

word_map = read_word_map(path=path, file_name=file_name_word_map)

word_count = read_sparse_npz(path=path, file_name=file_name_word_count)
assess_sparse_matrix(word_map=word_map, sparse_matrix=word_count)
#For this we should have used sklearn.Countvectorizer but we need assessment compatibility

tf_idf = read_sparse_npz(path=path, file_name=file_name_tf_idf)
assess_sparse_matrix(word_map=word_map, sparse_matrix=tf_idf)
#For this we should have used TfidfVectorizer but we need assessment compatibility

print_with_tms("start fit model")
model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)
print_with_tms("finished fit model")

i_obama = find_index_by_name(df=all_data_df, name='Barack Obama')
print(f"i_obama {i_obama}")
find_and_print_neighbours(df=all_data_df,
                          model=model,
                          word_count=word_count,
                          i=i_obama,
                          n_neighbors=10)

#Test top_words:
print_with_tms("\nTest top_words\n")
obama_words = top_words(
    df=all_data_df,
    name="Barack Obama",
    word_count=word_count,
    word_map=word_map
)
print(obama_words.head(10))
barrio_words = top_words(
    df=all_data_df,
    name="Francisco Barrio",
    word_count=word_count,
    word_map=word_map
)
print(barrio_words.head(10))
print(f"type(obama_words) {type(obama_words)}")

combined_words = (
    obama_words.merge(
        barrio_words,
        on="word",
        how="inner",
        suffixes=("_Obama", "_Barrio")
    )
    .rename(columns={
        "count_Obama": "Obama",
        "count_Barrio": "Barrio"
    })
    .sort_values("Obama", ascending=False)
    .reset_index(drop=True)
)

print(combined_words.head(10))


