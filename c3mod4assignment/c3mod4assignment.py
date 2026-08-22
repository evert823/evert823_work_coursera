'''
Almost copy of previous assignment except we're adding an L2 penalty
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

def point_7(df, word):
    filtered_df = df[df[word] > 0].copy()
    print(f"{len(filtered_df)} reviews contained the word {word}")

def create_np_matrix(df, columnnames):
    '''
    Convert df, columns indicated by parameter columnnames, to numpy array (2D matrix)
    If there are N data points and D features then return a NxD matrix
    '''
    feature_matrix = df[columnnames].values
    print_with_tms(f"Created feature matrix with shape {feature_matrix.shape}")
    return feature_matrix

def probability_from_score(z: float):
    score = 1 / (1 + (np.e ** (-z)))
    return score

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
    my_indicator = 1.0 if my_y == 1 else 0
    myscore = score_matrix[i]
    term_l = np.log(1 + np.exp(myscore * -1)) * -1

    result = ( (my_indicator - 1) * myscore ) + term_l
    return result

def print_stuff(iteration_nr, gradient_norm, log_l):
    s = f"iteration_nr {iteration_nr} gradient_norm {gradient_norm}"
    s += f" log_likelyhood {log_l}"
    print(s)

def gradient_with_L2(gradient, w_current, l2penalty):
    '''
    for entries 1 - D-1 subtract 2 * l2penalty * w_current[j]
    '''
    gradient_l2 = np.copy(gradient)
    gradient_l2[1:] -= 2 * l2penalty * w_current[1:]
    return gradient_l2

def l2term(w_current, l2penalty):
    '''
    sum of squares of coefficients w
    0-th coefficient w[0] excluded
    times l2penalty
    '''
    l2norm = np.sum(w_current[1:] ** 2)
    result = l2penalty * l2norm
    return result

def gradient_ascent_algorithm(w_init,
                              H, Y,
                              epsilon, stepsize,
                              max_iter, l2penalty):
    D = H.shape[1]
    iteration_nr = 0
    w_current = np.copy(w_init)
    gradient_norm = 100.0 #dummy large value to have it defined 1st iteration
    while iteration_nr < max_iter and gradient_norm > epsilon:
        score_matrix = np.matmul(H, w_current)
        P_class_1_by_data_point = probabilities_from_score_matrix(score_matrix=score_matrix)
        error = compute_error(Y=Y, P=P_class_1_by_data_point)
        gradient = np.matmul(H.T, error)

        gradient_l2 = gradient_with_L2(gradient=gradient,
                                       w_current=w_current,
                                       l2penalty=l2penalty)

        gradient_norm = np.linalg.norm(gradient_l2)
        log_l = compute_log_likelyhood(Y=Y, score_matrix=score_matrix)
        log_l -= l2term(w_current=w_current, l2penalty=l2penalty)
        if iteration_nr % 10 == 0:
            print_stuff(iteration_nr, gradient_norm, log_l)
        w_new = gradient_ascent_upd_w(D=D, gradient=gradient_l2,
                            w_current=w_current,
                            stepsize=stepsize)
        w_current = np.copy(w_new)
        iteration_nr += 1
    return w_current

def gradient_ascent_upd_w(D, gradient,
                          w_current, stepsize):
    w_new = np.copy(w_current)
    for j in range(D):
        partial_j = gradient[j]
        w_new[j] = w_new[j] + ( partial_j * stepsize )

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

def point_15(Y_predicted):
    count1 = 0
    countm1 = 0
    for i in range(Y_predicted.shape[0]):
        if Y_predicted[i] == 1:
            count1 += 1
        else:
            countm1 += 1
    print(f"Positive {count1} negative {countm1}")

def assess_relevance_of_words(w, significant_words):
    a = [(z, word) for z, word in zip(w[1:], significant_words)]
    a = sorted(a, key=lambda x:x[0], reverse=True)
    print(f"Ten most positive words {a[:10]}")
    print(f"Ten most negative words {a[-10:]}")



path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course03", "module04", "data")
file_name_inp = "amazon_baby_subset.csv"
all_data_df = read_data(path=path, file_name=file_name_inp, dtype_dict=amazon_baby_dtype_dict())
all_data_df['review_clean'] = all_data_df['review'].apply(remove_punctuation)
assess_dataframe(all_data_df)

significant_words = get_significant_words(path=path)
all_data_df = add_wordcounts_to_df(df=all_data_df, significant_words=significant_words)

all_data_df['ones'] = 1
columnnames = ['ones'] + significant_words

#Use index files for train and validation split (to reproduce a particular split)
with open(os.path.join(path, "module-4-assignment-train-idx.json")) as f:
    train_idx = json.load(f)
print(len(train_idx))
train_data_df = all_data_df.iloc[train_idx]
with open(os.path.join(path, "module-4-assignment-validation-idx.json")) as f:
    val_idx = json.load(f)
print(len(val_idx))
val_data_df = all_data_df.iloc[val_idx]

#For point 6:
H_train = create_np_matrix(df=train_data_df, columnnames=columnnames)
Y_train = create_np_matrix(df=train_data_df, columnnames=['sentiment'])
H_val = create_np_matrix(df=val_data_df, columnnames=columnnames)
Y_val = create_np_matrix(df=val_data_df, columnnames=['sentiment'])

#For point 12

#Below I just left the code from previous module and used H_train instead of H

w_init = np.zeros(H_train.shape[1], dtype=float)
w_optimized = gradient_ascent_algorithm(w_init=w_init,
                        H=H_train, Y=Y_train,
                        epsilon=1e-7, stepsize=1e-7,
                        max_iter = 301, l2penalty=1e-3)

print(f"w_optimized {w_optimized}")

Y_predicted = predict_class(H=H_train, w=w_optimized)
print(f"Y_predicted {Y_predicted.shape}")

point_15(Y_predicted=Y_predicted)
accuracy = compute_accuracy(Y=Y_train, Y_predicted=Y_predicted)
print(f"accuracy {accuracy}")

assess_relevance_of_words(w=w_optimized, significant_words=significant_words)

