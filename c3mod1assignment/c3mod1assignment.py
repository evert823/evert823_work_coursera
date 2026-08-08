import pandas as pd
import os
import string
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def amazon_baby_dtype_dict():
    dtype_dict = {'name':str, 'review':str, 'rating':int}
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

def point_8(logreg: LogisticRegression):
    print(logreg.coef_.shape)
    count_lt0 = 0
    count_eq0 = 0
    count_gt0 = 0
    for i in range(logreg.coef_.shape[1]):
        if logreg.coef_[0, i] > 0:
            count_gt0 += 1
        if logreg.coef_[0, i] == 0:
            count_eq0 += 1
        if logreg.coef_[0, i] < 0:
            count_lt0 += 1
    print(f"count_lt0 {count_lt0} count_eq0 {count_eq0} count_gt0 {count_gt0}")
        

path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course03", "module01", "data")
file_name_inp = "amazon_baby.csv"

all_data_df = read_data(path=path, file_name=file_name_inp, dtype_dict=amazon_baby_dtype_dict())
total_observations = len(all_data_df)
#assess_dataframe(all_data_df)
all_data_df = all_data_df[all_data_df['rating'] != 3]
total_observations_neq_3 = len(all_data_df)
all_data_df['review'] = all_data_df['review'].fillna('')
all_data_df['review_clean'] = all_data_df['review'].apply(remove_punctuation)

all_data_df['sentiment'] = all_data_df['rating'].apply(lambda rating : +1 if rating > 3 else -1)

print("Done loading and transforming full data file")

#Use index files for train and test split (to reproduce a particular split)
with open(os.path.join(path, "module-2-assignment-train-idx.json")) as f:
    train_idx = json.load(f)
print(len(train_idx))
train_df = all_data_df.iloc[train_idx]

with open(os.path.join(path, "module-2-assignment-test-idx.json")) as f:
    test_idx = json.load(f)
print(len(test_idx))
test_df = all_data_df.iloc[test_idx]

number_of_train_observations = len(train_df)
number_of_test_observations = len(test_df)
assert number_of_train_observations + number_of_test_observations == total_observations_neq_3

print("Done train test split")

#Muller Guido pg. 337
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train_df['review_clean'])
test_matrix = vectorizer.transform(test_df['review_clean'])
print("Done word matrix")

#Muller Guido pg. 59
logreg = LogisticRegression(max_iter=1000)
logreg.fit(train_matrix, train_df['sentiment'])
print("Done train logreg")

point_8(logreg)
