import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
import sys
sys.path.append("../c3mod5assignment")
from tree_binary_classifier import TreeBinaryClassifier

def print_with_tms(message):
    mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{mytimestamp}|{message}")

def read_data(path=".\\", file_name="data.csv", dtype_dict=None):
    mydata = pd.read_csv(os.path.join(path, file_name), dtype=dtype_dict)
    return mydata

def lending_club_dtype_dict():
    string_columns = [
        "term", "grade", "sub_grade", "emp_title", "emp_length",
        "home_ownership", "is_inc_v", "issue_d", "loan_status",
        "pymnt_plan", "url", "desc", "purpose", "title", "zip_code",
        "addr_state", "earliest_cr_line", "initial_list_status",
        "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d",
        "final_d", "status"
    ]

    integer_columns = [
        "id", "member_id", "loan_amnt", "funded_amnt", "policy_code",
        "not_compliant", "inactive_loans", "bad_loans", "emp_length_num",
        "delinq_2yrs_zero", "pub_rec_zero", "collections_12_mths_zero",
        "short_emp", "last_delinq_none", "last_record_none",
        "last_major_derog_none"
    ]

    float_columns = [
        "funded_amnt_inv", "int_rate", "installment", "annual_inc",
        "dti", "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq",
        "mths_since_last_record", "open_acc", "pub_rec", "revol_bal",
        "revol_util", "total_acc", "out_prncp", "out_prncp_inv",
        "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
        "total_rec_int", "total_rec_late_fee", "recoveries",
        "collection_recovery_fee", "last_pymnt_amnt",
        "collections_12_mths_ex_med", "mths_since_last_major_derog",
        "grade_num", "sub_grade_num", "payment_inc_ratio"
    ]

    return {
        **{column: "string" for column in string_columns},
        **{column: "Int64" for column in integer_columns},
        **{column: "float64" for column in float_columns},
    }

def assess_dataframe(df):
    print_with_tms(f"rowcount {df.shape[0]} colcount {df.shape[1]}")
    print_with_tms(f"dtypes {dict(df.dtypes)}")
    print_with_tms(f"columns\n{df.columns.tolist()}")
    if PRINTSTUFF == True:
        print_with_tms("First 5 rows:")
        print_with_tms(df.head())

def define_x_features():
    features = ['grade',                     # grade of the loan
                'term',                      # the term of the loan
                'home_ownership',            # home_ownership status: own, mortgage or rent
                'emp_length',                # number of years of employment (categorical format)
            ]
    return features

def prepare_data(df, x_features, y_features):
    df2 = df.copy()
    df2['safe_loans'] = df2['bad_loans'].apply(lambda x : +1 if x==0 else -1)
    df2 = df2.drop(columns=["bad_loans"])
    return df2[x_features + y_features]

def apply_one_hot_encoding(df):
    '''
    The result of one-ho encoding is that multiclass categorical features are replaced with binary features
    that contain only 0 or 1 as values
    '''
    df2 = pd.get_dummies(df, dtype=int)
    return df2

def create_np_matrix(df, columnnames):
    '''
    Convert df, columns indicated by parameter columnnames, to numpy array (2D matrix)
    If there are N data points and D features then return a NxD matrix
    '''
    feature_matrix = df[columnnames].values
    print_with_tms(f"Created feature matrix with shape {feature_matrix.shape}")
    return feature_matrix

PRINTSTUFF = False
print_with_tms("Start script")
path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course03", "module06", "data")

file_name_inp = "lending-club-data.csv"
all_data_df = read_data(path=path, file_name=file_name_inp, dtype_dict=lending_club_dtype_dict())

x_features = define_x_features()
y_features = ['safe_loans']

all_data_df = prepare_data(df=all_data_df, x_features=x_features, y_features=y_features)
all_data_df = apply_one_hot_encoding(df=all_data_df)
x_features_enc = [cn for cn in all_data_df.columns.to_list() if cn not in y_features]
assess_dataframe(all_data_df)

#Use index files for train and val split (to reproduce a particular split)
with open(os.path.join(path, "module-6-assignment-train-idx.json")) as f:
    train_idx = json.load(f)
train_data_df = all_data_df.iloc[train_idx]
with open(os.path.join(path, "module-6-assignment-validation-idx.json")) as f:
    val_idx = json.load(f)
val_data_df = all_data_df.iloc[val_idx]
if PRINTSTUFF == True:
    print_with_tms(f"total {len(all_data_df)} train {len(train_data_df)} val {len(val_data_df)}")

X_train = create_np_matrix(df=train_data_df, columnnames=x_features_enc)
Y_train = create_np_matrix(df=train_data_df, columnnames=y_features)

N = X_train.shape[0]

tbc_old = TreeBinaryClassifier()
tbc_old.max_depth = 6
tbc_old.min_node_size = 0
tbc_old.error_reduction_threshold = -1.0

if PRINTSTUFF == True:
    tbc_old.verbose = True
tbc_old.fit(X=X_train, Y=Y_train)

tbc_new = TreeBinaryClassifier()
tbc_new.max_depth = 6
tbc_new.min_node_size = 100
tbc_new.error_reduction_threshold = 0.0

if PRINTSTUFF == True:
    tbc_new.verbose = True
tbc_new.fit(X=X_train, Y=Y_train)

X_val = create_np_matrix(df=val_data_df, columnnames=x_features_enc)
Y_val = create_np_matrix(df=val_data_df, columnnames=y_features)

Y_val_predicted_old = tbc_old.predict(X=X_val)
accuracy_val_old = tbc_old.accuracy_on_dataset(X=X_val, Y=Y_val)
print_with_tms(f"classification error tbc_old {1 - accuracy_val_old}")


Y_val_predicted_new = tbc_new.predict(X=X_val)
accuracy_val_new = tbc_new.accuracy_on_dataset(X=X_val, Y=Y_val)
print_with_tms(f"classification error tbc_new {1 - accuracy_val_new}")

#For point 16
print("For point 16")
print(f"Y_val_predicted_old[0]\n{Y_val_predicted_old[0]}")
print(f"Y_val_predicted_new[0]\n{Y_val_predicted_new[0]}")
print(f"Nog 1x de hierbij horende fieatures : {x_features_enc}")

#For point 22
print("For point 22")
tbc_d2 = TreeBinaryClassifier()
tbc_d2.max_depth = 2
tbc_d2.min_node_size = 0
tbc_d2.error_reduction_threshold = -1.0

if PRINTSTUFF == True:
    tbc_d2.verbose = True
tbc_d2.fit(X=X_train, Y=Y_train)

tbc_d6 = TreeBinaryClassifier()
tbc_d6.max_depth = 6
tbc_d6.min_node_size = 0
tbc_d6.error_reduction_threshold = -1.0

if PRINTSTUFF == True:
    tbc_d6.verbose = True
tbc_d6.fit(X=X_train, Y=Y_train)

tbc_d14 = TreeBinaryClassifier()
tbc_d14.max_depth = 14
tbc_d14.min_node_size = 0
tbc_d14.error_reduction_threshold = -1.0

if PRINTSTUFF == True:
    tbc_d14.verbose = True
tbc_d14.fit(X=X_train, Y=Y_train)

accuracy_d2_train = tbc_d2.accuracy_on_dataset(X=X_train, Y=Y_train)
print_with_tms(f"classification error train d2 {1 - accuracy_d2_train}")
accuracy_d2_val = tbc_d2.accuracy_on_dataset(X=X_val, Y=Y_val)
print_with_tms(f"classification error val d2 {1 - accuracy_d2_val}")
accuracy_d6_train = tbc_d6.accuracy_on_dataset(X=X_train, Y=Y_train)
print_with_tms(f"classification error train d6 {1 - accuracy_d6_train}")
accuracy_d6_val = tbc_d6.accuracy_on_dataset(X=X_val, Y=Y_val)
print_with_tms(f"classification error val d6 {1 - accuracy_d6_val}")
accuracy_d14_train = tbc_d14.accuracy_on_dataset(X=X_train, Y=Y_train)
print_with_tms(f"classification error train d14 {1 - accuracy_d14_train}")
accuracy_d14_val = tbc_d14.accuracy_on_dataset(X=X_val, Y=Y_val)
print_with_tms(f"classification error val d14 {1 - accuracy_d14_val}")

#For point 25 and 26
print("For point 25 and 26")
l2 = tbc_d2.number_of_leaves()
l6 = tbc_d6.number_of_leaves()
l14 = tbc_d14.number_of_leaves()
print(f"l2 {l2} l6 {l6} l14 {l14}")

#For point 27
print("For point 27")

tbc_model4 = TreeBinaryClassifier()
tbc_model4.max_depth = 6
tbc_model4.min_node_size = 0
tbc_model4.error_reduction_threshold = -1.0

if PRINTSTUFF == True:
    tbc_model4.verbose = True
tbc_model4.fit(X=X_train, Y=Y_train)

tbc_model5 = TreeBinaryClassifier()
tbc_model5.max_depth = 6
tbc_model5.min_node_size = 0
tbc_model5.error_reduction_threshold = 0.0

if PRINTSTUFF == True:
    tbc_model5.verbose = True
tbc_model5.fit(X=X_train, Y=Y_train)

tbc_model6 = TreeBinaryClassifier()
tbc_model6.max_depth = 6
tbc_model6.min_node_size = 0
tbc_model6.error_reduction_threshold = 5.0

if PRINTSTUFF == True:
    tbc_model6.verbose = True
tbc_model6.fit(X=X_train, Y=Y_train)

accuracy_model4_train = tbc_model4.accuracy_on_dataset(X=X_train, Y=Y_train)
print_with_tms(f"classification error train model4 {1 - accuracy_model4_train}")
accuracy_model4_val = tbc_model4.accuracy_on_dataset(X=X_val, Y=Y_val)
print_with_tms(f"classification error val model4 {1 - accuracy_model4_val}")
accuracy_model5_train = tbc_model5.accuracy_on_dataset(X=X_train, Y=Y_train)
print_with_tms(f"classification error train model5 {1 - accuracy_model5_train}")
accuracy_model5_val = tbc_model5.accuracy_on_dataset(X=X_val, Y=Y_val)
print_with_tms(f"classification error val model5 {1 - accuracy_model5_val}")
accuracy_model6_train = tbc_model6.accuracy_on_dataset(X=X_train, Y=Y_train)
print_with_tms(f"classification error train model6 {1 - accuracy_model6_train}")
accuracy_model6_val = tbc_model6.accuracy_on_dataset(X=X_val, Y=Y_val)
print_with_tms(f"classification error val model6 {1 - accuracy_model6_val}")

#For point 29
print("For point 29")
lm4 = tbc_model4.number_of_leaves()
lm5 = tbc_model5.number_of_leaves()
lm6 = tbc_model6.number_of_leaves()
print(f"lm4 {lm4} lm5 {lm5} lm6 {lm6}")
