import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from datetime import datetime
import os
import json
import graphviz

def print_with_tms(message):
    mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{mytimestamp}|{message}")

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

def define_x_features():
    features = ['grade',                     # grade of the loan
                'sub_grade',                 # sub-grade of the loan
                'short_emp',                 # one year or less of employment
                'emp_length_num',            # number of years of employment
                'home_ownership',            # home_ownership status: own, mortgage or rent
                'dti',                       # debt to income ratio
                'purpose',                   # the purpose of the loan
                'term',                      # the term of the loan
                'last_delinq_none',          # has borrower had a delinquincy
                'last_major_derog_none',     # has borrower had 90 day or worse rating
                'revol_util',                # percent of available credit being used
                'total_rec_late_fee',        # total late fees received to day
            ]
    return features

def create_np_matrix(df, columnnames):
    '''
    Convert df, columns indicated by parameter columnnames, to numpy array (2D matrix)
    If there are N data points and D features then return a NxD matrix
    '''
    feature_matrix = df[columnnames].values
    print_with_tms(f"Created feature matrix with shape {feature_matrix.shape}")
    return feature_matrix

PRINTSTUFF = False

def prepare_data(df, x_features, y_features):
    df2 = df.copy()
    df2['safe_loans'] = df2['bad_loans'].apply(lambda x : +1 if x==0 else -1)
    df2 = df2.drop(columns=["bad_loans"])
    return df2[x_features + y_features]

def apply_one_hot_encoding(df):
    df2 = pd.get_dummies(df, dtype=int)
    return df2

def create_np_matrix(df, columnnames):
    '''
    Convert df, columns indicated by parameter columnnames, to numpy array (2D matrix)
    If there are N data points and D features then return a NxD matrix
    '''
    feature_matrix = df[columnnames].values
    if PRINTSTUFF == True:
        print_with_tms(f"Created feature matrix with shape {feature_matrix.shape}")
    return feature_matrix

def visualize_treemodel(tree, class_names, feature_names):
    output_dir = ".\\output"
    os.makedirs(output_dir, exist_ok=True)

    dot_file = os.path.join(output_dir, "mytree.dot")
    export_graphviz(decision_tree=tree,
                    out_file=dot_file,
                    class_names=class_names,
                    feature_names=feature_names,
                    impurity=False,
                    filled=True)
    graph = graphviz.Source.from_file(dot_file)
    graph.render(
        filename=os.path.join(output_dir, "mytree"),
        format="png",
        cleanup=True
    )

def count_false_positives_false_negatives(tree, X, Y):
    '''
    Input: and ALREADY TRAINED tree decision model tree
    '''
    false_pos_count = 0
    false_neg_count = 0
    predictions = tree.predict(X=X)
    p_flat = np.asarray(predictions).reshape(-1)
    y_flat = np.asarray(Y).reshape(-1)
    numberofpredictions = len(p_flat)
    numberofknownvalues = len(y_flat)
    assert numberofknownvalues == numberofpredictions
    for i in range(numberofpredictions):
        if p_flat[i] == -1 and y_flat[i] == 1:
            false_neg_count += 1
        if p_flat[i] == 1 and y_flat[i] == -1:
            false_pos_count += 1

    return false_pos_count, false_neg_count

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

#Use index files for train and validation split (to reproduce a particular split)
with open(os.path.join(path, "module-5-assignment-1-train-idx.json")) as f:
    train_idx = json.load(f)
train_data_df = all_data_df.iloc[train_idx]
with open(os.path.join(path, "module-5-assignment-1-validation-idx.json")) as f:
    val_idx = json.load(f)
val_data_df = all_data_df.iloc[val_idx]
if PRINTSTUFF == True:
    print_with_tms(f"total {len(all_data_df)} train {len(train_data_df)} val {len(val_data_df)}")


#Section to build the very small samples BEGIN
sample_risky_df = val_data_df.loc[val_data_df["safe_loans"] == -1].head(2).copy()
sample_safe_df = val_data_df.loc[val_data_df["safe_loans"] == 1].head(2).copy()
sample_df = pd.concat([sample_risky_df, sample_safe_df], ignore_index=True)
#Section to build the very small samples END


X_train = create_np_matrix(df=train_data_df, columnnames=x_features_enc)
Y_train = create_np_matrix(df=train_data_df, columnnames=y_features)
X_val = create_np_matrix(df=val_data_df, columnnames=x_features_enc)
Y_val = create_np_matrix(df=val_data_df, columnnames=y_features)
X_sample = create_np_matrix(df=sample_df, columnnames=x_features_enc)
Y_sample = create_np_matrix(df=sample_df, columnnames=y_features)
print_with_tms("start tree.fit")

tree_depth_6 = DecisionTreeClassifier(random_state=0, max_depth=6)
tree_depth_6.fit(X=X_train, y=Y_train)
print_with_tms("finished tree.fit")
accuracy_train_depth_6 = tree_depth_6.score(X=X_train, y=Y_train)
accuracy_val_depth_6 = tree_depth_6.score(X=X_val, y=Y_val)
print(f"accuracy_train_depth_6 {accuracy_train_depth_6} accuracy_val_depth_6 {accuracy_val_depth_6}")

tree_depth_2 = DecisionTreeClassifier(random_state=0, max_depth=2)
tree_depth_2.fit(X=X_train, y=Y_train)
print_with_tms("finished tree.fit")
accuracy_train_depth_2 = tree_depth_2.score(X=X_train, y=Y_train)
accuracy_val_depth_2 = tree_depth_2.score(X=X_val, y=Y_val)
print(f"accuracy_train_depth_2 {accuracy_train_depth_2} accuracy_val_depth_2 {accuracy_val_depth_2}")

visualize_treemodel(tree=tree_depth_2,
                    class_names=["unsafe", "safe"],
                    feature_names=x_features_enc)

predictions = tree_depth_6.predict(X=X_sample)
print_with_tms(f"predictions\n{predictions}")
print_with_tms(f"input\n{sample_df['safe_loans']}")
accuracy_sample_depth_6 = tree_depth_6.score(X=X_sample, y=Y_sample)
print(f"accuracy_sample_depth_6 {accuracy_sample_depth_6}")

predictions_proba = tree_depth_6.predict_proba(X=X_sample)
print(predictions_proba)

#Point 14
print_with_tms("Going to point 14")
predictions_proba = tree_depth_2.predict_proba(X=X_sample)
print(predictions_proba)

#Point 14a
print_with_tms("Going to point 14a")
print(sample_df.iloc[1][['grade_A', 'total_rec_late_fee']])

#Point 18
print_with_tms("Going to point 18")
tree_depth_10 = DecisionTreeClassifier(random_state=0, max_depth=10)
tree_depth_10.fit(X=X_train, y=Y_train)
print_with_tms("finished tree.fit")
accuracy_train_depth_10 = tree_depth_10.score(X=X_train, y=Y_train)
accuracy_val_depth_10 = tree_depth_10.score(X=X_val, y=Y_val)
print(f"accuracy_train_depth_10 {accuracy_train_depth_10} accuracy_val_depth_10 {accuracy_val_depth_10}")

#Point 20
print_with_tms("Going to point 20")
false_pos_count, false_neg_count = count_false_positives_false_negatives(tree=tree_depth_6, X=X_val, Y=Y_val)
cost = ( false_neg_count * 10000 ) + ( false_pos_count * 20000 )
print_with_tms(f"false_pos_count {false_pos_count} false_neg_count {false_neg_count} cost {cost}")
