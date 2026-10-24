import os
from datetime import datetime
import pandas as pd
import numpy as np
import json
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

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


def bipartition(cluster, maxiter=400, num_runs=4, seed=None):
    '''
    Method largely provided by course
    cluster: should be a dictionary containing the following keys
                * dataframe: original dataframe
                * matrix:    same data, in matrix format
                * centroid:  centroid for this particular cluster
    '''
    
    data_matrix = cluster['matrix']
    dataframe   = cluster['dataframe']
    
    # Run k-means on the data matrix with k=2. We use scikit-learn here to simplify workflow.
    kmeans_model = KMeans(n_clusters=2, max_iter=maxiter, n_init=num_runs, random_state=seed)    
    kmeans_model.fit(data_matrix)
    centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_
    
    # Divide the data matrix into two parts using the cluster assignments.
    data_matrix_left_child, data_matrix_right_child = data_matrix[cluster_assignment==0], \
                                                      data_matrix[cluster_assignment==1]
    
    # Divide the dataframe into two parts, again using the cluster assignments.
    # Here we eliminated use of SFrame
    dataframe_left_child = dataframe.loc[cluster_assignment == 0]
    dataframe_right_child = dataframe.loc[cluster_assignment == 1]
    
    # Package relevant variables for the child clusters
    cluster_left_child  = {'matrix': data_matrix_left_child,
                           'dataframe': dataframe_left_child,
                           'centroid': centroids[0]}
    cluster_right_child = {'matrix': data_matrix_right_child,
                           'dataframe': dataframe_right_child,
                           'centroid': centroids[1]}
    
    return (cluster_left_child, cluster_right_child)


def display_single_tf_idf_cluster(cluster, map_index_to_word, filepath):
    '''
    Method largely provided by course
    map_index_to_word: SFrame specifying the mapping betweeen words and column indices
    '''
    index_to_word = {
        index: word for word, index in map_index_to_word.items()
    }

    file2 = open(filepath, 'w')
    
    wiki_subset   = cluster['dataframe']
    tf_idf_subset = cluster['matrix']
    centroid      = cluster['centroid']

    # Print top 5 words with largest TF-IDF weights in the cluster
    idx = centroid.argsort()[::-1]
    for i in range(5):
        file2.write('{0:s}:{1:.3f}'.format(index_to_word[idx[i]], centroid[idx[i]]))
        file2.write('\n')
    file2.write('\n')
    
    # Compute distances from the centroid to all data points in the cluster.
    distances = pairwise_distances(tf_idf_subset, [centroid], metric='euclidean').flatten()
    # compute nearest neighbors of the centroid within the cluster.
    nearest_neighbors = distances.argsort()
    # For 8 nearest neighbors, print the title as well as first 180 characters of text.
    # Wrap the text at 80-character mark.
    for i in range(8):
        text = ' '.join(wiki_subset.iloc[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
        file2.write('* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki_subset.iloc[nearest_neighbors[i]]['name'],
              distances[nearest_neighbors[i]], text[:90], text[90:180] if len(text) > 90 else ''))
        file2.write('\n')
    file2.write('\n')
    file2.close()


PRINTSTUFF = False

print_with_tms("script started")

path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course04", "module06", "data")
output_dir = os.path.join(os.path.dirname(__file__), "output")

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

tf_idf_norm = normalize(X=tf_idf)
assess_sparse_matrix(word_map=word_map, sparse_matrix=tf_idf_norm)

wiki_data = {'matrix': tf_idf_norm, 'dataframe': all_data_df} # no 'centroid' for the root cluster
left_child, right_child = bipartition(cluster=wiki_data,
                                      maxiter=100,
                                      num_runs=6,
                                      seed=1)
display_single_tf_idf_cluster(cluster=left_child,
                              map_index_to_word=word_map,
                              filepath=os.path.join(output_dir, "left_child_01.txt"))
display_single_tf_idf_cluster(cluster=right_child,
                              map_index_to_word=word_map,
                              filepath=os.path.join(output_dir, "right_child_01.txt"))


#athletes = left_child
#non_athletes = right_child
#By coincidence got swapped compared to course
athletes = right_child
non_athletes = left_child

left_child_athletes, right_child_athletes = bipartition(cluster=athletes,
                                                        maxiter=100,
                                                        num_runs=6,
                                                        seed=1)

display_single_tf_idf_cluster(cluster=left_child_athletes,
                              map_index_to_word=word_map,
                              filepath=os.path.join(output_dir, "left_child_athletes_01.txt"))
display_single_tf_idf_cluster(cluster=right_child_athletes,
                              map_index_to_word=word_map,
                              filepath=os.path.join(output_dir, "right_child_athletes_01.txt"))

baseball            = right_child_athletes
ice_hockey_football = left_child_athletes

left_child_ihs, right_child_ihs = bipartition(cluster=ice_hockey_football,
                                                        maxiter=100,
                                                        num_runs=6,
                                                        seed=1)
display_single_tf_idf_cluster(cluster=left_child_ihs,
                              map_index_to_word=word_map,
                              filepath=os.path.join(output_dir, "left_child_ihs_01.txt"))
display_single_tf_idf_cluster(cluster=right_child_ihs,
                              map_index_to_word=word_map,
                              filepath=os.path.join(output_dir, "right_child_ihs_01.txt"))



print_with_tms("script finished")
