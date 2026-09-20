'''
Almost copy of script from course 3 module 3 but we're making it stochastic
This assignment is without L2 penalty
'''
import os
import numpy as np
import pandas as pd
import string
import json
from datetime import datetime

def print_with_tms(message):
    mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{mytimestamp}|{message}")

def amazon_baby_dtype_dict():
    dtype_dict = {'name':str, 'review':str, 'rating':int, 'sentiment':int}
    return dtype_dict

def read_data(path=".\\", file_name="data.csv", dtype_dict=None):
    mydata = pd.read_csv(os.path.join(path, file_name), dtype=dtype_dict)
    return mydata

def assess_dataframe(df):
    print(f"rowcount {df.shape[0]} colcount {df.shape[1]}")
    print(f"dtypes {dict(df.dtypes)}")
    if PRINTSTUFF == True:
        print("First 5 rows:")
        print(df.head())

def remove_punctuation(text):
    if pd.isna(text):  # Handle NaN values
        return text
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def get_significant_words(path):
    with open(os.path.join(path, "important_words.json")) as f:
        significant_words = json.load(f)
    return significant_words

def add_wordcounts_to_df(df, significant_words):
    new_columns = {}
    for i in range(len(significant_words)):
        word = significant_words[i]
        new_columns[word] = df['review_clean'].apply(lambda text: wordcount_in_text(word=word, text=text))
        if i % 20 == 0:
            print_with_tms(f"word no. {i} {word} finished wordcount all data points for this word")

    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    print_with_tms(f"finished pd.concat")
    #cross_check_word_count(df=df, word="found")
    return df

def wordcount_in_text(word: str, text: str) -> int:
    '''
    How often does the word appear in the text?
    E.g. word nice text "The soup was nice but the pasta was not so nice" result 2
    '''
    if pd.isna(text):  # Handle NaN values
        return 0
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    word_lower = word.lower()
    # Split text into words and count occurrences
    words = text_lower.split()
    return words.count(word_lower)

def cross_check_word_count(df, word):
    for i in range(len(df)):
        mytext = df.iloc[i]['review_clean']
        mycount_1 = df.iloc[i][word]
        mycount_2 = wordcount_in_text(text=mytext, word=word)
        print(f"mycount_1 {mycount_1} mycount_2 {mycount_2}")
        assert mycount_1 == mycount_2

def create_np_matrix(df, columnnames):
    '''
    Convert df, columns indicated by parameter columnnames, to numpy array (2D matrix)
    If there are N data points and D features then return a NxD matrix
    '''
    feature_matrix = df[columnnames].values
    print_with_tms(f"Created feature matrix with shape {feature_matrix.shape}")
    return feature_matrix

def probabilities_from_score_matrix(score_matrix):
    return 1.0 / (1.0 + np.exp(-score_matrix))

def compute_error(Y, P):
    '''
    Y has shape (N, 1, )
    P has shape (N,)
    Here we need Indicator[y_i = +1] - P[i]
    Result must be of shape (N,)
    '''
    y_flat = np.asarray(Y).reshape(-1)
    p_flat = np.asarray(P).reshape(-1)
    return (y_flat == 1).astype(float) - p_flat


def compute_log_likelyhood(Y, score_matrix):
    N = Y.shape[0]
    total = 0.0
    for i in range(N):
        term = compute_log_likelyhood_term(i=i,
                                           Y=Y,
                                           score_matrix=score_matrix)
        total += term
    return total

def compute_log_likelyhood_term(i, Y, score_matrix):
    my_y = Y[i, 0]
    my_indicator = 1.0 if my_y == 1 else 0.0
    myscore = score_matrix[i]
    term_l = np.log(1 + np.exp(myscore * -1)) * -1

    result = ( (my_indicator - 1) * myscore ) + term_l
    return result

def print_stuff(iteration_nr, gradient_norm, log_l):
    if PRINTSTUFF == False:
        return
    s = f"iteration_nr {iteration_nr} gradient_norm {gradient_norm}"
    s += f" log_likelyhood {log_l}"
    print(s)

def gradient_ascent_algorithm_stochastic_batch(w_current,
                                         H, Y,
                                         batch_size,
                                         batchnr):
    '''
    Stochastic Gradient Ascent algorithm - the code to be executed for each batch
    '''
    N = H.shape[0]
    start_i = batchnr * batch_size
    next_start_i = np.minimum(start_i + batch_size, N)
    H_batch = H[start_i:next_start_i,:]
    Y_batch = Y[start_i:next_start_i,:]

    score_matrix = np.matmul(H_batch, w_current)
    P_class_1_by_data_point = probabilities_from_score_matrix(score_matrix=score_matrix)
    error = compute_error(Y=Y_batch, P=P_class_1_by_data_point)
    gradient = np.matmul(H_batch.T, error)
    log_l_batch = compute_log_likelyhood(Y=Y_batch, score_matrix=score_matrix)

    return gradient, log_l_batch

def avg_log_l_batch(log_l_batch_outcomes, k):
    last_k = log_l_batch_outcomes[-k:]
    return sum(last_k) / len(last_k)

def check_values(k, batch_size, N, epsilon):
    if not isinstance(k, int):
        raise TypeError("k must be an integer")
    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if not isinstance(epsilon, float):
        raise TypeError("epsilon must be a float")

    if k <= 0:
        raise ValueError("k must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size > N:
        raise ValueError("batch_size cannot exceed N")
    if epsilon < 0 or epsilon >= 100.0:
        raise ValueError("pls choose a proper value for epsilon")

def gradient_ascent_algorithm_stochastic(w_init,
                                         H, Y,
                                         batch_size,
                                         epsilon, stepsize,
                                         max_iter, k):
    '''
    Stochastic Gradient Ascent algorithm without L2 regularization
    '''
    N = H.shape[0]
    D = H.shape[1]

    check_values(k=k, batch_size=batch_size, N=N, epsilon=epsilon)

    avg_log_l = 100.0
    prev_avg_log_l = 0.0

    w_current = np.copy(w_init)
    log_l_batch_outcomes = []

    iteration_nr = 0
    while iteration_nr < max_iter and np.abs(avg_log_l - prev_avg_log_l) >= epsilon:
        print_with_tms(f"iteration_nr {iteration_nr}")
        #Here, an iteration is a whole data pass (N rows) and usually we do one or few
        #While a batch is a subset of the data

        shuffled_idx = np.random.permutation(N)
        H_shuffled = H[shuffled_idx]
        Y_shuffled = Y[shuffled_idx]

        batchnr = 0
        while batchnr * batch_size < N:
            gradient, log_l_batch = gradient_ascent_algorithm_stochastic_batch(w_current=w_current,
                                                H=H_shuffled, Y=Y_shuffled,
                                                batch_size=batch_size,
                                                batchnr=batchnr)

            log_l_batch_outcomes.append(log_l_batch)
            prev_avg_log_l = avg_log_l
            avg_log_l = avg_log_l_batch(log_l_batch_outcomes=log_l_batch_outcomes,
                                        k=k)

            w_new = gradient_ascent_upd_w(D=D, gradient=gradient,
                                w_current=w_current,
                                stepsize=stepsize, batch_size=batch_size)
            w_current = np.copy(w_new)

            if batch_size >= 2000 or batchnr % 2000 == 0 or PRINTSTUFF == True:
                print_with_tms(f"batchnr {batchnr} avg_log_l {avg_log_l}")
            batchnr += 1

        iteration_nr += 1
    return w_current


def gradient_ascent_upd_w(D, gradient,
                          w_current, stepsize, batch_size):
    '''
    Here we divide the term by batch_size before adding the term
    to the previous coefficient value
    '''
    w_new = np.copy(w_current)
    for j in range(D):
        partial_j = gradient[j]
        w_new[j] = w_new[j] + ( partial_j * stepsize / batch_size)

    return w_new

def predict_class(H, w):
    score_matrix = np.matmul(H, w)
    Y_predicted = np.where(score_matrix > 0, 1, -1)
    return Y_predicted

def compute_accuracy(Y, Y_predicted):
    y_flat = np.asarray(Y).reshape(-1)
    y_predicted_flat = np.asarray(Y_predicted).reshape(-1)
    N = len(y_flat)
    assert N == len(y_predicted_flat)

    correctcount = 0
    for i in range(N):
        iscorrect = False
        if y_flat[i] == 1 and y_predicted_flat[i] == 1:
            iscorrect = True
        if y_flat[i] == 1.0 and y_predicted_flat[i] == 1.0:
            iscorrect = True
        if y_flat[i] <= 0 and y_predicted_flat[i] <= 0:
            iscorrect = True
        if iscorrect == True:
            correctcount += 1

    return correctcount / N


PRINTSTUFF = False
print_with_tms("script started")
path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course03", "module10", "data")
file_name_inp = "amazon_baby_subset.csv"
all_data_df = read_data(path=path, file_name=file_name_inp, dtype_dict=amazon_baby_dtype_dict())
all_data_df['review_clean'] = all_data_df['review'].apply(remove_punctuation)
assess_dataframe(all_data_df)

significant_words = get_significant_words(path=path)
all_data_df = add_wordcounts_to_df(df=all_data_df, significant_words=significant_words)

all_data_df['ones'] = 1
columnnames = ['ones'] + significant_words

#Use index files for train and validation split (to reproduce a particular split)
with open(os.path.join(path, "module-10-assignment-train-idx.json")) as f:
    train_idx = json.load(f)
print(len(train_idx))
train_data_df = all_data_df.iloc[train_idx]
with open(os.path.join(path, "module-10-assignment-validation-idx.json")) as f:
    val_idx = json.load(f)
print(len(val_idx))
val_data_df = all_data_df.iloc[val_idx]

H_train = create_np_matrix(df=train_data_df, columnnames=columnnames)
Y_train = create_np_matrix(df=train_data_df, columnnames=['sentiment'])
H_val = create_np_matrix(df=val_data_df, columnnames=columnnames)
Y_val = create_np_matrix(df=val_data_df, columnnames=['sentiment'])

w_init = np.zeros(H_train.shape[1], dtype=float)
w_optimized = gradient_ascent_algorithm_stochastic(w_init=w_init,
                        H=H_train, Y=Y_train,
                        batch_size=1,
                        epsilon=0.0, stepsize=5e-1,
                        max_iter = 10, k=1)
print_with_tms(f"w_optimized \n{w_optimized}")

