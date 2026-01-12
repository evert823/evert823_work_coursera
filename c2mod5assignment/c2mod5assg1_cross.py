import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def house_data_dtype_dict():
    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
                'sqft_living15':float, 'grade':int, 'yr_renovated':int,
                'price':float, 'bedrooms':float, 'zipcode':str,
                'long':float, 'sqft_lot15':float, 'sqft_living':float,
                'floors':str, 'condition':int, 'lat':float, 'date':str,
                'sqft_basement':int, 'yr_built':int, 'id':str,
                'sqft_lot':int, 'view':int}
    return dtype_dict

def read_house_data(path=".\\", file_name="data.csv", dtype_dict=None):
    mydata = pd.read_csv(path + file_name, dtype=dtype_dict)
    return mydata

def assess_dataframe(df):
    print(f"rowcount {df.shape[0]} colcount {df.shape[1]}")
    print(f"dtypes {dict(df.dtypes)}")
    #print("First 5 rows:")
    #print(df.head())

def add_terms_of_polynomial(df, feature_name, degree):
    #Add columns polynomial terms as columns to dataframe till/including degree
    df_out = df.copy()

    for i in range(2, degree + 1):
        fn = f"{feature_name}_pwr_{i}"
        df_out[fn] = df_out[feature_name] ** i

    return df_out

def train_on_df(df, l2_penalty, verbose=False):
    if verbose == True:
        print(f"train_on_df {len(df)} L2 {l2_penalty}")
    degree = 15
    x_feature_names = ['sqft_living']
    for i in range(2, degree + 1):
        x_feature_names.append(f'sqft_living_pwr_{i}')
    y_feature_names = ['price']

    #Create the dataframe with terms of polynomial of given degree
    polynomial_df = add_terms_of_polynomial(df=df, feature_name='sqft_living', degree=degree)

    #Create and fit a model
    y = polynomial_df[y_feature_names].values
    X = polynomial_df[x_feature_names].values
    model = make_pipeline(StandardScaler(), Ridge(alpha=l2_penalty))
    model.fit(X, y)
    a = model.named_steps['ridge']
    w = np.hstack((a.intercept_.reshape(1, 1), a.coef_))
    if verbose == True:
        print(f"w with StandardScaler: {w} shape : {w.shape}")

    return model

def evaluate_on_df(df, model, verbose=False):
    if verbose == True:
        print(f"evaluate_on_df {len(df)}")
    degree = 15
    x_feature_names = ['sqft_living']
    for i in range(2, degree + 1):
        x_feature_names.append(f'sqft_living_pwr_{i}')
    y_feature_names = ['price']

    #Create the dataframe with terms of polynomial of given degree
    polynomial_df = add_terms_of_polynomial(df=df, feature_name='sqft_living', degree=degree)
    y = polynomial_df[y_feature_names].values
    X = polynomial_df[x_feature_names].values

    y_pred = model.predict(X)
    rs = (y - y_pred) * (y - y_pred)
    rss = rs.sum()
    return rss

def k_fold_cross_validation(df, k, l2_penalty, verbose=False):
    n = len(df)
    rss_total = 0.0
    for i in range(k):
        start = n * i // k
        end = n * (i + 1) // k
        df_train1 = shuffled_all_df.iloc[0:start] 
        df_valid = shuffled_all_df.iloc[start:end]
        df_train2 = shuffled_all_df.iloc[end:n]
        df_train = pd.concat([df_train1, df_train2], ignore_index=True)
        assert len(df_train1) + len(df_train2) == len(df_train)
        assert len(df_train) + len(df_valid) == n
        mylinreg = train_on_df(df=df_train, l2_penalty=l2_penalty)
        rss = evaluate_on_df(df=df_valid, model=mylinreg)
        rss_total += rss
        if verbose == True:
            print(f"rss {rss}")
    rss_avg = rss_total / k
    return rss_avg

path = r"C:\Users\Evert Jan\courseradatascience\course02\module05\data\\"
shuffled_all_df = read_house_data(path=path, file_name="wk3_kc_house_train_valid_shuffled.csv", dtype_dict=house_data_dtype_dict())
assess_dataframe(shuffled_all_df)#19396
n = len(shuffled_all_df)

a = np.logspace(3, 9, num=13)
#a = np.logspace(0, 9, num=13)
for l2 in a:
    rss = k_fold_cross_validation(df=shuffled_all_df, k=10, l2_penalty=l2)
    print(f"rss from k-fold cross validation L2 {l2} rss {rss}")

mylinreg = train_on_df(df=shuffled_all_df, l2_penalty=1000)
test_df = read_house_data(path=path, file_name="wk3_kc_house_test_data.csv", dtype_dict=house_data_dtype_dict())
rss = evaluate_on_df(df=test_df, model=mylinreg)
print(f"rss for L2 = {1000} on testdata {rss}")
