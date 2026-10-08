import numpy as np
import os
from scipy import sparse
from datetime import datetime
import json

def print_with_tms(message):
    mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{mytimestamp}|{message}")


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

def load_precomputed_clusters(path, file_name="kmeans-arrays.npz"):
    file_path = os.path.join(path, file_name)
    arrays = np.load(file_path, allow_pickle=False)
    return arrays


PRINTSTUFF = False

path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course04", "module03", "data")

file_name_word_map = "people_wiki_map_index_to_word.json"
file_name_tf_idf = "people_wiki_tf_idf.npz"
file_name_precomputed_clusters = "kmeans-arrays.npz"

word_map = read_word_map(path=path, file_name=file_name_word_map)
tf_idf = read_sparse_npz(path=path, file_name=file_name_tf_idf)
assess_sparse_matrix(word_map=word_map, sparse_matrix=tf_idf)

arrays = load_precomputed_clusters(path=path, file_name=file_name_precomputed_clusters)
print(type(arrays))
print("Available arrays:", arrays.files)

for name in arrays.files:
    array = arrays[name]
    print(f"{name}: {array.nbytes / 1024**2:.2f} MB")
    print(type(array))
    print(array.shape)
#Copy this in-memory is A LOT so we need a more economic approach
arrays.close()
