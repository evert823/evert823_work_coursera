'''
Re-use of c3mod2assignment.py possible
'''
import pandas as pd
import os
import string
import json
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

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
