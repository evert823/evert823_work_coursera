'''
Re-use of c3mod2assignment.py possible
'''
import numpy as np
import pandas as pd
import os
import string
import json
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
      accuracy_score,
      confusion_matrix,
      precision_score,
      recall_score
     )
import matplotlib.pyplot as plt

def print_with_tms(message):
    mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{mytimestamp}|{message}")

def amazon_baby_dtype_dict():
    dtype_dict = {'name':str, 'review':str, 'rating':int}
    return dtype_dict

def read_data(path=".\\", file_name="data.csv", dtype_dict=None):
    mydata = pd.read_csv(os.path.join(path, file_name), dtype=dtype_dict)
    return mydata

def assess_dataframe(df):
    print_with_tms(f"rowcount {df.shape[0]} colcount {df.shape[1]}")
    print_with_tms(f"dtypes {dict(df.dtypes)}")
    print_with_tms(f"columns\n{df.columns.tolist()}")
    if PRINTSTUFF == True:
        print_with_tms("First 5 rows:")
        print_with_tms(df.head())

def remove_punctuation(text):
    if pd.isna(text):  # Handle NaN values
        return ''
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def cost_mistakes(confmatrix):
    cost_false_positive = 100
    cost_false_negative = 1
    false_positive_count = confmatrix[0, 1] #true negative predicted positive
    false_negative_count = confmatrix[1, 0] #true positive predicted negative
    return (cost_false_positive * false_positive_count) + (cost_false_negative * false_negative_count)

def nice_print_confusion_matrix(logreg, confmatrix):
    for i, target_label in enumerate(logreg.classes_):
        for j, predicted_label in enumerate(logreg.classes_):
            print('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, confmatrix[i,j]))

def predict_using_threshold(model: LogisticRegression, p_threshold, X):
    '''
    y_pred2 will have 3 columns:
    - probability -1 per data point
    - probability 1 per data point
    - predicted label based on threshold
    '''
    i_1 = list(model.classes_).index(1) #probably i_1==1
    y_pred = model.predict_proba(X=X)

    predicted_label = np.where(
        y_pred[:, i_1]  >= p_threshold,
        1,
        -1
    )
    y_pred2 = np.hstack((y_pred, predicted_label.reshape(-1, 1)))

    if PRINTSTUFF == True:
        print_with_tms(f"type(y_pred2) {type(y_pred2)}")
        print_with_tms(f"y_pred2.shape {y_pred2.shape}")
        with np.printoptions(suppress=True, precision=8):
            print_with_tms(f"y_pred2[:6,:]\n{y_pred2[:6,:]}")

    return y_pred2

def plot_pr_curve(precision_all, recall_all, png_file_name):
    plt.figure(figsize=(8, 6))
    plt.plot(precision_all, recall_all, '-o', markersize=3)

    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "output",
        png_file_name
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print_with_tms(f"Saved plot to {output_path}")

def try_threshold(p_threshold, logreg, test_matrix, test_df):
    y_pred_wt = predict_using_threshold(model=logreg, p_threshold=p_threshold, X=test_matrix)

    cmat_wt = confusion_matrix(y_true=test_df['sentiment'],
                            y_pred=y_pred_wt[:, 2].astype(int),
                            labels=logreg.classes_)
    precision_wt = precision_score(y_true=test_df['sentiment'],
                                y_pred=y_pred_wt[:, 2].astype(int))
    recall_wt = recall_score(y_true=test_df['sentiment'],
                            y_pred=y_pred_wt[:, 2].astype(int))
    if PRINTSTUFF == True:
        print(f"cmat_wt\n{cmat_wt}")
        print_with_tms(f"with threshold {p_threshold} precision_wt {precision_wt} recall_wt {recall_wt}")
    return precision_wt, recall_wt


print_with_tms("Script started")
path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course03", "module09", "data")
file_name_inp = "amazon_baby.csv"

PRINTSTUFF = False

all_data_df = read_data(path=path, file_name=file_name_inp, dtype_dict=amazon_baby_dtype_dict())
all_data_df = all_data_df[all_data_df['rating'] != 3].copy()
total_observations_neq_3 = len(all_data_df)
all_data_df['sentiment'] = all_data_df['rating'].apply(lambda rating : +1 if rating > 3 else -1)
all_data_df['review_clean'] = all_data_df['review'].apply(remove_punctuation)
assess_dataframe(all_data_df)

#Use index files for train and test split (to reproduce a particular split)
with open(os.path.join(path, "module-9-assignment-train-idx.json")) as f:
    train_idx = json.load(f)
print(len(train_idx))
train_df = all_data_df.iloc[train_idx]

with open(os.path.join(path, "module-9-assignment-test-idx.json")) as f:
    test_idx = json.load(f)
print(len(test_idx))
test_df = all_data_df.iloc[test_idx]

number_of_train_observations = len(train_df)
number_of_test_observations = len(test_df)
assert number_of_train_observations + number_of_test_observations == total_observations_neq_3

#Muller Guido pg. 337
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train_df['review_clean'])
print_with_tms("Created train_matrix")
test_matrix = vectorizer.transform(test_df['review_clean'])
print_with_tms("Created test_matrix")

#Muller Guido pg. 59
logreg = LogisticRegression(max_iter=1000)
logreg.fit(train_matrix, train_df['sentiment'])
print_with_tms("logreg.fit done")

test_predictions = logreg.predict(X=test_matrix)
test_accuracy = accuracy_score(y_true=test_df['sentiment'], y_pred=test_predictions)
print_with_tms(f"test_accuracy: {test_accuracy}")

majority_clf_accuracy = len(test_df[test_df['sentiment'] == 1])/len(test_df)
print_with_tms(f"majority_clf_accuracy: {majority_clf_accuracy}")

cmat = confusion_matrix(y_true=test_df['sentiment'],
                        y_pred=test_predictions,
                        labels=logreg.classes_)
print_with_tms(f"cmat \n{cmat}")
nice_print_confusion_matrix(logreg=logreg, confmatrix=cmat)
cost = cost_mistakes(confmatrix=cmat)
print(f"cost {cost}")

precision = precision_score(y_true=test_df['sentiment'],
                            y_pred=test_predictions)
recall = recall_score(y_true=test_df['sentiment'],
                      y_pred=test_predictions)
print_with_tms(f"Precision on test data {precision}")
print_with_tms(f"Recall on test data {recall}")

for p in [0.5, 0.9]:
    precision_wt, recall_wt = try_threshold(p_threshold=p,
                                            logreg=logreg,
                                            test_matrix=test_matrix,
                                            test_df=test_df)

threshold_values = np.linspace(0.5, 1, num=100)
precision_all = []
recall_all = []
for p in threshold_values:
    precision_wt, recall_wt = try_threshold(p_threshold=p,
                                            logreg=logreg,
                                            test_matrix=test_matrix,
                                            test_df=test_df)
    precision_all.append(precision_wt)
    recall_all.append(recall_wt)

plot_pr_curve(
    precision_all=precision_all,
    recall_all=recall_all,
    png_file_name="precision_recall_curve.png"
)


'''
Below we repeat the assessment of the model on testdata for only baby related products
'''
test_baby_only_df = test_df[
    test_df['name'].str.contains('baby', case=False, na=False)
].copy()
print_with_tms(f"Subset baby related only {len(test_baby_only_df)}")
test_baby_only_matrix = vectorizer.transform(test_baby_only_df['review_clean'])


precision_all = []
recall_all = []
for p in threshold_values:
    precision_wt, recall_wt = try_threshold(p_threshold=p,
                                            logreg=logreg,
                                            test_matrix=test_baby_only_matrix,
                                            test_df=test_baby_only_df)
    precision_all.append(precision_wt)
    recall_all.append(recall_wt)

plot_pr_curve(
    precision_all=precision_all,
    recall_all=recall_all,
    png_file_name="precision_recall_curve_baby_only.png"
)
