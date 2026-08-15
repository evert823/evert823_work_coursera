import pandas as pd
import os
import string
import json
import numpy as np
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

def probability_from_score(z: float):
    score = 1 / (1 + (np.e ** (-z)))
    return score

def point_8(logreg: LogisticRegression):
    coef = logreg.coef_[0]
    print(f"Coefficient shape: {coef.shape}")
    print(f"count_lt0: {np.sum(coef < 0)}")
    print(f"count_eq0: {np.sum(coef == 0)}")
    print(f"count_gt0: {np.sum(coef > 0)}")

def show_accuracy(complete_df):
    count_correct_prediction = np.sum(complete_df['sentiment'] == complete_df['predicted_sentiment'])
    accuracy = count_correct_prediction / len(complete_df)
    print(f"Correct predictions: {count_correct_prediction}")
    print(f"Total predictions: {len(complete_df)}")
    print(f"Accuracy: {accuracy:.4f}")


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


#till including point 11:
sample_df = test_df.iloc[10:13]
print(sample_df.iloc[0]['review'])
print(sample_df.iloc[1]['review'])
sample_matrix = vectorizer.transform(sample_df['review_clean'])
sample_df['score'] = logreg.decision_function(sample_matrix)
sample_df['p_1'] = sample_df['score'].apply(probability_from_score)

#sample_df['predicted_sentiment'] = sample_df['score'].apply(lambda score : +1 if score > 0 else -1)
sample_df['predicted_sentiment'] = logreg.predict(sample_matrix)

for i in range(len(sample_df)):
    print(sample_df.iloc[i])

#For point 13:
print("Going to point 13")
test_df['score'] = logreg.decision_function(test_matrix)
test_df['p_1'] = test_df['score'].apply(probability_from_score)
test_df['predicted_sentiment'] = logreg.predict(test_matrix)
top20_from_test_df = test_df.sort_values(['score', 'p_1'], ascending=False).iloc[:20]
for i in range(len(top20_from_test_df)):
    print(top20_from_test_df.iloc[i]['name'])

#For point 14:
print("Going to point 14")
bottom20_from_test_df = test_df.sort_values(['score', 'p_1'], ascending=True).iloc[:20]
for i in range(len(bottom20_from_test_df)):
    print(bottom20_from_test_df.iloc[i]['name'])

print("Going to point 15")
show_accuracy(complete_df=test_df)

print("Going to point 16")
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']

vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
#Recreate train_df and test_df
train_df = all_data_df.iloc[train_idx]
test_df = all_data_df.iloc[test_idx]

train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_df['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_df['review_clean'])
logreg_word_subset = LogisticRegression(max_iter=1000)
logreg_word_subset.fit(train_matrix_word_subset, train_df['sentiment'])

print("for point 18:")
point_8(logreg=logreg_word_subset)

print("for point 20:")
test_df['score'] = logreg_word_subset.decision_function(test_matrix_word_subset)
test_df['p_1'] = test_df['score'].apply(probability_from_score)
test_df['predicted_sentiment'] = logreg_word_subset.predict(test_matrix_word_subset)
show_accuracy(complete_df=test_df)

#For point 21
print("for point 21:")
majority_sentiment = train_df['sentiment'].value_counts().idxmax()
test_df['majority_prediction'] = majority_sentiment
majority_accuracy = np.sum(test_df['sentiment'] == test_df['majority_prediction']) / len(test_df)
print(f"Majority class baseline accuracy: {majority_accuracy:.4f}")
