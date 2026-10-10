import os
import pandas as pd
import numpy as np
from kmeans_cluster import KMeansCluster
from datetime import datetime
import json
from scipy import sparse
from sklearn.preprocessing import normalize

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

def assess_loaded_numpy_arrays(arrays):
    print(f"type(arrays) {type(arrays)}")
    print(f"arrays.files {arrays.files}")

def run_seeds(model, tf_idf_norm, use_kpp_method, default_seed_array, dummy=False):
    if dummy == True:
        return
    heterogeneity = {}
    for seed in default_seed_array:
        centroids, htgn = model.fit(X=tf_idf_norm, k=10, seed=seed,
                            epsilon=1e-8,max_iterations=400, use_kpp_method=use_kpp_method)
        heterogeneity[seed] = htgn
        labelcounts = model.report_label_per_data_point()
        print(labelcounts)
    print(f"heterogeneity {heterogeneity}")

def run_fit_multiple_init_one_k(model, tf_idf_norm, default_seed_array, dummy=False):
    if dummy == True:
        return
    best_centroids, best_heterogeneity, best_seed = model.fit_multiple_init_one_k(X=tf_idf_norm,
                                                        k=10,
                                                        seed_array=default_seed_array,
                                                        epsilon=1e-8,max_iterations=400,
                                                        use_kpp_method=True)
    print_with_tms(f"best_centroids {best_centroids} best_heterogeneity {best_heterogeneity}")
    labelcounts = model.report_label_per_data_point()
    print_with_tms(f"labelcounts :\n{labelcounts}")

def visualize_one_cluster(model, tf_idf_norm, centroids, word_map, all_data_df,
                          label, representative_indices):
    original_index = int(representative_indices[label, 0])

    if original_index == -1:
        print(f"Cluster {label} is empty.")
        return

    print(f"\nRepresentative data point for cluster {label}:")
    print(all_data_df.iloc[original_index])

    index_to_word = {
        column_index: word
        for word, column_index in word_map.items()
    }

    centroid = centroids[label]
    top_indices = np.argsort(centroid)[-5:][::-1]

    print("Top five words in the centroid:")
    for index in top_indices:
        print(index_to_word.get(int(index), "<unknown>"), centroid[index])


def cluster_visualization(model, tf_idf_norm, centroids, word_map, all_data_df):
    '''
    For each centroid determine the data point nearest to the centroid
    '''
    k = centroids.shape[0]
    D = centroids.shape[1]
    representative_datapoints, representative_indices = model.find_representative_data_points(X=tf_idf_norm, centroids=centroids)
    print(f"representative_datapoints.shape {representative_datapoints.shape}")
    print(f"representative_indices.shape {representative_indices.shape}")

    for label in range(k):
        visualize_one_cluster(model=model, tf_idf_norm=tf_idf_norm, centroids=centroids,
                              word_map=word_map, all_data_df=all_data_df,
                              label=label, representative_indices=representative_indices)

PRINTSTUFF = False

print_with_tms("script started")

path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course04", "module03", "data")

'''
The input file is comma separated and with text qualifier double quotes
'''
file_name_inp = "people_wiki.csv"
file_name_word_map = "people_wiki_map_index_to_word.json"
file_name_tf_idf = "people_wiki_tf_idf.npz"
file_name_precomputed_clusters = "kmeans-arrays.npz"

all_data_df = read_data(path=path, file_name=file_name_inp, dtype_dict=people_wiki_dtype_dict())
assess_dataframe(df=all_data_df)

word_map = read_word_map(path=path, file_name=file_name_word_map)
tf_idf = read_sparse_npz(path=path, file_name=file_name_tf_idf)
assess_sparse_matrix(word_map=word_map, sparse_matrix=tf_idf)
#For this we should have used TfidfVectorizer but we need assessment compatibility

tf_idf_norm = normalize(X=tf_idf)
assess_sparse_matrix(word_map=word_map, sparse_matrix=tf_idf_norm)

model = KMeansCluster()
if PRINTSTUFF == True:
    model.IsVerbose = True

centroids, htgn = model.fit(X=tf_idf_norm,
                      k=3,
                      seed=0,
                      epsilon=1e-8,
                      max_iterations=400)
print_with_tms(type(centroids))
print_with_tms(f"centroids\n{centroids}")

labelcounts = model.report_label_per_data_point()
print_with_tms(f"labelcounts :\n{labelcounts}")

default_seed_array = [0, 20000, 40000, 60000, 80000, 100000, 120000]

print_with_tms("Going to Beware of local minima")
run_seeds(model=model, tf_idf_norm=tf_idf_norm, use_kpp_method=False,
          default_seed_array=default_seed_array, dummy=True)

print_with_tms("k-means++ initialization")
run_seeds(model=model, tf_idf_norm=tf_idf_norm, use_kpp_method=True,
          default_seed_array=default_seed_array, dummy=True)

print_with_tms("Now we write and use the method fit_multiple_init_one_k")
run_fit_multiple_init_one_k(model=model, tf_idf_norm=tf_idf_norm,
                            default_seed_array=default_seed_array,
                            dummy=True)

print_with_tms("k-search")
output_dir = os.path.join(os.path.dirname(__file__), "output")

model.k_search(X=tf_idf_norm,
               k_array=[2, 10, 25, 50, 100],
               seed_array=default_seed_array,
               epsilon=1e-8,max_iterations=400,
               use_kpp_method=True,
               png_file_path=os.path.join(output_dir, "k_search.png"),
               log_file_path=os.path.join(output_dir, "k_search.log"),
               dummy=True)


print("\nNow we inspect the quality of our clusters\n")
centroids, best_heterogeneity, _ = model.fit_multiple_init_one_k(X=tf_idf_norm,
                                                    k=10,
                                                    seed_array=default_seed_array,
                                                    epsilon=1e-8,max_iterations=400,
                                                    use_kpp_method=True)
cluster_visualization(model=model, tf_idf_norm=tf_idf_norm, centroids=centroids,
                      word_map=word_map, all_data_df=all_data_df)
