import os
from datetime import datetime
import pandas as pd
import numpy as np
import json
from scipy import sparse
from sklearn.preprocessing import normalize
from em_model_hd import EMModelHD
from sklearn.cluster import KMeans
from em_utilities import diag

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

def initialize_by_kmeans(tf_idf):
    np.random.seed(5)
    num_clusters = 25

    # Use scikit-learn's k-means to simplify workflow
    kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1)
    kmeans_model.fit(tf_idf)
    centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_
    print_with_tms(f"cluster_assignment.shape {cluster_assignment.shape}")
    #cluster_assignment.shape (5000,)

    means = [centroid for centroid in centroids]

    num_docs = tf_idf.shape[0]
    weights = []

    cluster_counts = np.bincount(
        cluster_assignment,
        minlength=num_clusters
    )

    for i in range(num_clusters):
        # Compute the number of data points assigned to cluster i:
        num_assigned = cluster_counts[i]
        w = float(num_assigned) / num_docs
        weights.append(w)

    covs = []
    for i in range(num_clusters):
        member_rows = tf_idf[cluster_assignment==i]
        cov = (member_rows.multiply(member_rows) - 2*member_rows.dot(diag(means[i]))).sum(axis=0).A1 / member_rows.shape[0] \
            + means[i]**2
        cov[cov < 1e-8] = 1e-8
        covs.append(cov)

    return means, covs, weights

def visualize_EM_clusters(tf_idf, means, covs, map_index_to_word, output_dir, reportname="EM_clusters_report.txt"):
    index_to_word = {
        index: word for word, index in map_index_to_word.items()
    }
    outputpath = os.path.join(output_dir, reportname)
    file2 = open(outputpath, "w")

    file2.write('\n==========================================================\n')
    num_clusters = len(means)
    for c in range(num_clusters):
        file2.write('\nCluster {0:d}: Largest mean parameters in cluster '.format(c))
        file2.write('\n')
        file2.write('{0: <12}{1: <12}{2: <12}'.format('Word', 'Mean', 'Variance'))
        file2.write('\n')
        
        # The k'th element of sorted_word_ids should be the index of the word 
        # that has the k'th-largest value in the cluster mean. Hint: Use np.argsort().
        sorted_word_ids = np.argsort(a=means[c],axis=0)[::-1]

        for i in sorted_word_ids[:5]:
            file2.write('{0: <12}{1:<10.2e}{2:10.2e}'.format(index_to_word[i], 
                                                       means[c][i],
                                                       covs[c][i]))
            file2.write('\n')
    file2.write('\n===Quiz Question. Select all the topics that have a cluster in the model created above. [multiple choice]===\n')
    file2.close()
    print_with_tms(f"Report of EM clusters written to {outputpath}")


def random_initialization(means, tf_idf):
    np.random.seed(5)
    num_clusters = len(means)
    num_docs, num_words = tf_idf.shape

    random_means = []
    random_covs = []
    random_weights = []

    for k in range(num_clusters):
        
        # Create a numpy array of length num_words with random normally distributed values.
        # Use the standard univariate normal distribution (mean 0, variance 1).
        # YOUR CODE HERE
        mean = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=num_words
        )
        
        # Create a numpy array of length num_words with random values uniformly distributed between 1 and 5.
        # YOUR CODE HERE
        cov = np.random.uniform(
            low=1.0,
            high=5.0,
            size=num_words
        )

        # Initially give each cluster equal weight.
        # YOUR CODE HERE
        weight = 1.0 / num_clusters
        
        random_means.append(mean)
        random_covs.append(cov)
        random_weights.append(weight)

    return random_means, random_covs, random_weights




PRINTSTUFF = False

print_with_tms("script started")

path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course04", "module04", "data")

output_dir = os.path.join(os.path.dirname(__file__), "output")

'''
The input file is comma separated and with text qualifier double quotes
'''
file_name_inp = "people_wiki.csv"
file_name_word_map = "4_map_index_to_word.json"
file_name_tf_idf = "4_tf_idf.npz"

all_data_df = read_data(path=path, file_name=file_name_inp, dtype_dict=people_wiki_dtype_dict())
assess_dataframe(df=all_data_df)

word_map = read_word_map(path=path, file_name=file_name_word_map)
tf_idf = read_sparse_npz(path=path, file_name=file_name_tf_idf)
assess_sparse_matrix(word_map=word_map, sparse_matrix=tf_idf)
#For this we should have used TfidfVectorizer but we need assessment compatibility
#Apparently the course topped number of obeservations at 5000 in this tf idf matrix

tf_idf_norm = normalize(X=tf_idf)
assess_sparse_matrix(word_map=word_map, sparse_matrix=tf_idf_norm)

model = EMModelHD()
if PRINTSTUFF == True:
    model.IsVerbose = True

print_with_tms("initialize_by_kmeans BEGIN (THERE MAY BE A FEW WARNINGS NOW)")
means, covs, weights = initialize_by_kmeans(tf_idf=tf_idf_norm)
print_with_tms("initialize_by_kmeans DONE")

print(f"len(means) {len(means)} type(means) {type(means)}")
print(f"len(covs) {len(covs)} type(covs) {type(covs)}")
print(f"len(weights) {len(weights)} type(weights) {type(weights)}")

print_with_tms("EM_for_high_dimension BEGIN")
out = model.EM_for_high_dimension(data=tf_idf_norm,
                                  means=means, covs=covs, weights=weights,
                                  cov_smoothing=1e-10)
print_with_tms("EM_for_high_dimension DONE")
print(out['loglik']) # print history of log-likelihood over time

visualize_EM_clusters(tf_idf=tf_idf_norm,
                      means=out['means'],
                      covs=out['covs'],
                      map_index_to_word=word_map,
                      output_dir=output_dir)

print_with_tms("\nNow we try the same procedure with random initialization\n")
random_means, random_covs, random_weights = random_initialization(means=means, tf_idf=tf_idf_norm)

print_with_tms("EM_for_high_dimension BEGIN")
out_poor = model.EM_for_high_dimension(data=tf_idf_norm,
                                  means=random_means, covs=random_covs, weights=random_weights,
                                  cov_smoothing=1e-5)
print_with_tms("EM_for_high_dimension DONE")
print(out_poor['loglik']) # print history of log-likelihood over time

visualize_EM_clusters(tf_idf=tf_idf_norm,
                      means=out_poor['means'],
                      covs=out_poor['covs'],
                      map_index_to_word=word_map,
                      output_dir=output_dir,
                      reportname="EM_clusters_report_random_initialization.txt")
