from datetime import datetime
import os
import pandas as pd
import numpy as np
import json
from scipy import sparse
from lsh_model import LSHModel

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

PRINTSTUFF = False

print_with_tms("script started")

path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course04", "module02", "data")

'''
The input file is comma separated and with text qualifier double quotes
'''
file_name_inp = "people_wiki.csv"
file_name_word_map = "people_wiki_map_index_to_word.json"
file_name_tf_idf = "people_wiki_tf_idf.npz"

all_data_df = read_data(path=path, file_name=file_name_inp, dtype_dict=people_wiki_dtype_dict())
assess_dataframe(df=all_data_df)

word_map = read_word_map(path=path, file_name=file_name_word_map)

tf_idf = read_sparse_npz(path=path, file_name=file_name_tf_idf)
assess_sparse_matrix(word_map=word_map, sparse_matrix=tf_idf)
#For this we should have used TfidfVectorizer but we need assessment compatibility

model = LSHModel(h=16, r=1, L=1, random_seed=143)
model.IsVerbose = False

#For testing purpose
random_vectors = model.generate_random_vectors(dim=5)
print(f"type(random_vectors) {type(random_vectors)}")
print(f"random_vectors.shape {random_vectors.shape}")
print(f"random_vectors\n{random_vectors}")

model.fit(X=tf_idf)

for i in [0, 143]:
    try:
        print(f"model.table[{i}] {model.table[i]}")
    except:
        pass


i_obama = find_index_by_name(df=all_data_df, name='Barack Obama')
print(all_data_df.iloc[i_obama])
print(f"model.index_bits[i_obama] {model.index_bits[i_obama]}")
print(f"model.index_numbers[i_obama] {model.index_numbers[i_obama]}")

i_biden = find_index_by_name(df=all_data_df, name='Joe Biden')
print(all_data_df.iloc[i_biden])
print(f"model.index_bits[i_biden] {model.index_bits[i_biden]}")
print(f"model.index_numbers[i_biden] {model.index_numbers[i_biden]}")

i_hughjones = find_index_by_name(df=all_data_df, name='Wynn Normington Hugh-Jones')
print(all_data_df.iloc[i_hughjones])
print(f"model.index_bits[i_hughjones] {model.index_bits[i_hughjones]}")
print(f"model.index_numbers[i_hughjones] {model.index_numbers[i_hughjones]}")


print_with_tms("\n\n")

bin_obama = model.index_numbers[i_obama]
print(f"model.table[bin_obama] {model.table[bin_obama]}")
for i2 in model.table[bin_obama]:
    print(all_data_df.iloc[i2])
    print(f"model.index_bits[{i2}] {model.index_bits[i2]}")
    print(f"model.index_bits[{i_obama}] {model.index_bits[i_obama]}")

obama_tf_idf = tf_idf[35817,:]
biden_tf_idf = tf_idf[24478,:]
a = model.cosine_distance(x=obama_tf_idf, y=biden_tf_idf)
print(f"distance Obama Biden {a}")
for i2 in model.table[bin_obama]:
    if i2 != i_obama:
        doc_tf_idf = tf_idf[i2,:]
        a = model.cosine_distance(x=obama_tf_idf, y=doc_tf_idf)
        print(f"distance Obama other doc {a}")


model.search(X=tf_idf, i=i_obama)
