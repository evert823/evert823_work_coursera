from datetime import datetime
import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

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
    features = ['grade',                     # grade of the loan (categorical)
                'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
                'short_emp',                 # one year or less of employment
                'emp_length_num',            # number of years of employment
                'home_ownership',            # home_ownership status: own, mortgage or rent
                'dti',                       # debt to income ratio
                'purpose',                   # the purpose of the loan
                'payment_inc_ratio',         # ratio of the monthly payment to income
                'delinq_2yrs',               # number of delinquincies
                'delinq_2yrs_zero',          # no delinquincies in last 2 years
                'inq_last_6mths',            # number of creditor inquiries in last 6 months
                'last_delinq_none',          # has borrower had a delinquincy
                'last_major_derog_none',     # has borrower had 90 day or worse rating
                'open_acc',                  # number of open credit accounts
                'pub_rec',                   # number of derogatory public records
                'pub_rec_zero',              # no derogatory public records
                'revol_util',                # percent of available credit being used
                'total_rec_late_fee',        # total late fees received to day
                'int_rate',                  # interest rate of the loan
                'total_rec_int',             # interest received to date
                'annual_inc',                # annual income of borrower
                'funded_amnt',               # amount committed to the loan
                'funded_amnt_inv',           # amount committed by investors for the loan
                'installment',               # monthly payment owed by the borrower
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

def sample_from_validation_data(val_df):
    safe_loans = val_df.loc[val_df["safe_loans"] == 1].head(2)
    unsafe_loans = val_df.loc[val_df["safe_loans"] == -1].head(2)
    return pd.concat([safe_loans, unsafe_loans], axis=0)

def false_positives_false_negatives(X, Y, model):
    Y_pred = model.predict(X)
    Y_flat = np.asarray(Y).reshape(-1)
    Y_pred_flat = np.asarray(Y_pred).reshape(-1)
    false_positives = 0
    false_negatives = 0

    assert len(Y_flat) == len(Y_pred_flat)

    for i in range(len(Y_flat)):
        if Y_flat[i] == 1 and Y_pred_flat[i] == -1:
            false_negatives += 1
        if Y_flat[i] == -1 and Y_pred_flat[i] == 1:
            false_positives += 1

    return false_positives, false_negatives

def iteration(X_train, Y_train, X_val, Y_val, n_estimators):
    print_with_tms(f"Start training model with n_estimators={n_estimators}")
    mymodel = GradientBoostingClassifier(max_depth=6,
                                         n_estimators=n_estimators,
                                         #learning_rate=0.3,
                                         random_state=0
                                         )
    mymodel.fit(X=X_train,y=Y_train)
    accrcy_train = mymodel.score(X=X_train,
                               y=Y_train)
    accrcy_val = mymodel.score(X=X_val,
                               y=Y_val)
    cl_err_train = 1 - accrcy_train
    cl_err_val = 1 - accrcy_val
    print_with_tms(f"Finished training model with n_estimators={n_estimators}")
    return mymodel, cl_err_train, cl_err_val

def assess_results(n_estimators_grid,
                   cl_err_train_grid,
                   cl_err_val_grid,
                   output_dir):

    for i in range(len(n_estimators_grid)):
        print(f"i {i} n_est {n_estimators_grid[i]} err train {cl_err_train_grid[i]} error val {cl_err_val_grid[i]}")

    plt.figure(figsize=(9, 6))
    plt.plot(
        n_estimators_grid,
        cl_err_train_grid,
        marker="o",
        linewidth=2,
        label="error_train"
    )
    plt.plot(
        n_estimators_grid,
        cl_err_val_grid,
        marker="s",
        linewidth=2,
        label="error_val"
    )

    plt.title("Gradient Boosting Performance")
    plt.xlabel("Number of estimators")
    plt.ylabel("Score")
    plt.xticks(n_estimators_grid)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "gradient_boosting_performance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print_with_tms(f"Plot saved to {plot_path}")    

PRINTSTUFF = False
print_with_tms("Start script")
path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course03", "module08", "data")
output_dir = os.path.join(".", "output")

x_features = define_x_features()
y_features = ['safe_loans']

file_name_inp = "lending-club-data.csv"
all_data_df = read_data(path=path, file_name=file_name_inp, dtype_dict=lending_club_dtype_dict())
all_data_df = prepare_data(df=all_data_df, x_features=x_features, y_features=y_features)
all_data_df = all_data_df.dropna() # here only deletes 29 data points - 122578 remain

all_data_noenc_df = all_data_df.copy()

all_data_df = apply_one_hot_encoding(df=all_data_df)
x_features_enc = [cn for cn in all_data_df.columns.to_list() if cn not in y_features]

assess_dataframe(all_data_df)

#Use index files for train and val split (to reproduce a particular split)
#Important decision point: this happens AFTER dropna
with open(os.path.join(path, "module-8-assignment-1-train-idx.json")) as f:
    train_idx = json.load(f)
train_data_df = all_data_df.iloc[train_idx]
with open(os.path.join(path, "module-8-assignment-1-validation-idx.json")) as f:
    val_idx = json.load(f)
val_data_df = all_data_df.iloc[val_idx]
print_with_tms(f"total {len(all_data_df)} train {len(train_data_df)} val {len(val_data_df)}")

#Muller Guido page 90 - 94

X_train = create_np_matrix(df=train_data_df, columnnames=x_features_enc)
Y_train = create_np_matrix(df=train_data_df, columnnames=y_features)
X_val = create_np_matrix(df=val_data_df, columnnames=x_features_enc)
Y_val = create_np_matrix(df=val_data_df, columnnames=y_features)

model_5 = GradientBoostingClassifier(max_depth=6,
                                     n_estimators=5,
                                     #learning_rate=0.3,
                                    )
model_5.fit(X=X_train,y=Y_train)

#Point 10 and 11
print("Point 10 and 11")
sample_df = sample_from_validation_data(val_df=val_data_df)
X_sample = create_np_matrix(df=sample_df, columnnames=x_features_enc)
Y_sample = create_np_matrix(df=sample_df, columnnames=y_features)
Y_pred_sample = model_5.predict(X=X_sample)
print_with_tms(f"Y_sample\n{Y_sample}")
print_with_tms(f"Y_pred_sample\n{Y_pred_sample}")

#Point 12
print("Point 12")
Y_prob_sample = model_5.predict_proba(X=X_sample)
print_with_tms(f"Y_prob_sample\n{Y_prob_sample}")

score_sample = model_5.score(X=X_sample,
                        y=Y_sample)
print(f"score on sample {score_sample}")

#Point 13
print("Point 13")
score_val = model_5.score(X=X_val,
                        y=Y_val)
print(f"score on validation {score_val}")

#Point 14 & 15
print("Point 14 & 15")
false_positives, false_negatives = false_positives_false_negatives(X=X_val,
                                                                   Y=Y_val,
                                                                   model=model_5)
print(f"false_positives {false_positives} false_negatives {false_negatives}")

#Point 16
print("Point 16")

cost_model_5 = (10000.0 * false_negatives) + (20000.0 * false_positives)
print_with_tms(f"cost_model_5 {cost_model_5}")

#Point 17-18-19
print("Point 17-18-19")
Y_prob_val = model_5.predict_proba(X=X_val)
val_data_noenc_df = all_data_noenc_df.iloc[val_idx]
grade_probabilities_df = pd.DataFrame({
    "grade": val_data_noenc_df["grade"].to_numpy(),
    "probability_-1": Y_prob_val[:, 0],
    "probability_1": Y_prob_val[:, 1],
})
grade_probabilities_df = grade_probabilities_df.sort_values(
    by="probability_1",
    ascending=False
).reset_index(drop=True)
print(f"top 5\n{grade_probabilities_df.iloc[:5]}")
print(f"bottom 5\n{grade_probabilities_df.iloc[-5:]}")

n_estimators_grid = [10, 50, 100, 200, 500]
model_grid = []
cl_err_train_grid = []
cl_err_val_grid = []

for n in n_estimators_grid:
    mymodel, cl_err_train, cl_err_val = iteration(X_train=X_train,
                                                  Y_train=Y_train,
                                                  X_val=X_val,
                                                  Y_val=Y_val,
                                                  n_estimators=n)
    model_grid.append(mymodel)
    cl_err_train_grid.append(cl_err_train)
    cl_err_val_grid.append(cl_err_val)

assess_results(n_estimators_grid=n_estimators_grid,
               cl_err_train_grid=cl_err_train_grid,
               cl_err_val_grid=cl_err_val_grid,
               output_dir=output_dir)
