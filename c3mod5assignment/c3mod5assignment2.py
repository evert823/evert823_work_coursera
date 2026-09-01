import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
from tree_binary_classifier import TreeBinaryClassifier
graphviz_path = r"C:\Program Files\Graphviz\bin"
if graphviz_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + graphviz_path
import graphviz

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

PRINTSTUFF = True

print_with_tms("Start script")
path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course03", "module05", "data")

file_name_inp = "lending-club-data.csv"
all_data_df = read_data(path=path, file_name=file_name_inp, dtype_dict=lending_club_dtype_dict())

x_features = define_x_features()
y_features = ['safe_loans']

all_data_df = prepare_data(df=all_data_df, x_features=x_features, y_features=y_features)
all_data_df = apply_one_hot_encoding(df=all_data_df)
x_features_enc = [cn for cn in all_data_df.columns.to_list() if cn not in y_features]
assess_dataframe(all_data_df)

#Use index files for train and test split (to reproduce a particular split)
with open(os.path.join(path, "module-5-assignment-2-train-idx.json")) as f:
    train_idx = json.load(f)
train_data_df = all_data_df.iloc[train_idx]
with open(os.path.join(path, "module-5-assignment-2-test-idx.json")) as f:
    test_idx = json.load(f)
test_data_df = all_data_df.iloc[test_idx]
if PRINTSTUFF == True:
    print_with_tms(f"total {len(all_data_df)} train {len(train_data_df)} test {len(test_data_df)}")

X_train = create_np_matrix(df=train_data_df, columnnames=x_features_enc)
Y_train = create_np_matrix(df=train_data_df, columnnames=y_features)

N = X_train.shape[0]

#We can calculate majority class and probability given a node
tbc = TreeBinaryClassifier()
tbc.max_depth = 6
tbc.min_node_size = 10
tbc.error_reduction_threshold = 0.01
if PRINTSTUFF == True:
    tbc.verbose = True
tbc.fit(X=X_train, Y=Y_train)

X_test = create_np_matrix(df=test_data_df, columnnames=x_features_enc)
Y_test_predicted = tbc.predict(X=X_test[:10])
Y_test = create_np_matrix(df=test_data_df, columnnames=y_features)

print(f"Y_test[:10]\n{Y_test[:10]}")
print(f"Y_test_predicted\n{Y_test_predicted}")
print(f"Nog 1x de hierbij horende fieatures : {x_features_enc}")


#Point 16 & 17
print("Now going to point 16 and 17")
accuracy_test = tbc.accuracy_on_dataset(X=X_test, Y=Y_test)
print_with_tms(f"accuracy_test {accuracy_test} classification error {1 - accuracy_test}")

#Point 18
print("Now going to point 18")
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "lending_club_tree.dot")
tbc.save_tree_as_dot(filename=output_file)
print("Tree saved as lending_club_tree.dot")


try:
    with open(output_file, 'r') as f:
        dot_content = f.read()
    graph = graphviz.Source(dot_content)
    graph.render(filename=os.path.join(output_dir, "lending_club_tree"), format='png', cleanup=True)
    print("Tree rendered as lending_club_tree.png")
except Exception as e:
    print(f"Error rendering tree: {e}")
