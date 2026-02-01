import pandas as pd
import numpy as np
from math import log, sqrt
from sklearn import linear_model

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
    print("First 5 rows:")
    print(df.head())

def add_columns(df):
    df['floors'] = df['floors'].astype(float)  # Convert floors to float
    df['sqft_living_sqrt'] = df['sqft_living'].apply(sqrt)
    df['sqft_lot_sqrt'] = df['sqft_lot'].apply(sqrt)
    df['bedrooms_square'] = df['bedrooms']*df['bedrooms']
    df['floors_square'] = df['floors']*df['floors']
    return df

def read_fit(df, l1_penalty=5e2):
    print(f"read_fit with l1_penalty = {l1_penalty}")
    df = add_columns(df=df)
    mymodel = linear_model.Lasso(alpha=l1_penalty)
    mymodel.fit(df[all_features], df['price'])
    a = np.array([mymodel.intercept_])
    w = np.hstack((a, mymodel.coef_))
    print(w)
    nz = np.count_nonzero(w)
    print(f"#nonzero coefficients {nz} - this includes the intercept")
    return mymodel, nz

def compute_rss(df, model):
    df = add_columns(df=df)
    y_pred = model.predict(df[all_features])
    rs = (df['price'] - y_pred) * (df['price'] - y_pred)
    rss = rs.sum()
    return rss

path = r"C:\Users\Evert Jan\courseradatascience\course02\module06\data\\"

#house_data_all_df = read_house_data(path=path, file_name="kc_house_data.csv", dtype_dict=house_data_dtype_dict())
house_data_train_df = read_house_data(path=path, file_name="wk3_kc_house_train_data.csv", dtype_dict=house_data_dtype_dict())
house_data_test_df = read_house_data(path=path, file_name="wk3_kc_house_test_data.csv", dtype_dict=house_data_dtype_dict())
house_data_valid_df = read_house_data(path=path, file_name="wk3_kc_house_valid_data.csv", dtype_dict=house_data_dtype_dict())

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

allpenalties = np.logspace(1, 7, num=13)
rss_list = []
for l1_penalty in allpenalties:
    mymodel, nz = read_fit(df=house_data_train_df, l1_penalty=l1_penalty)
    rss = compute_rss(df=house_data_valid_df, model=mymodel)
    rss_list.append(rss)

for i in range(len(rss_list)):
    print(f"l1_penalty {allpenalties[i]} rss {rss_list[i]}")

best_penalty = 10
mymodel, nz = read_fit(df=house_data_train_df, l1_penalty=best_penalty)
rss = compute_rss(df=house_data_test_df, model=mymodel)
print(f"l1_penalty {best_penalty} rss on testdata {rss}")

#From here we answer q9 and further
allpenalties2 = np.logspace(1, 4, num=20)
nz_list = []
for l1_penalty in allpenalties2:
    mymodel, nz = read_fit(df=house_data_train_df, l1_penalty=l1_penalty)
    nz_list.append(nz)
for i in range(len(nz_list)):
    print(f"l1_penalty {allpenalties2[i]} nz {nz_list[i]}")

l1_penalty_min = allpenalties2[7]
l1_penalty_max = allpenalties2[9]

allpenalties3 = np.linspace(l1_penalty_min,l1_penalty_max,20)
rss_list = []
nz_list = []
for l1_penalty in allpenalties3:
    mymodel, nz = read_fit(df=house_data_train_df, l1_penalty=l1_penalty)
    nz_list.append(nz)
    rss = compute_rss(df=house_data_valid_df, model=mymodel)
    rss_list.append(rss)

for i in range(len(rss_list)):
    print(f"l1_penalty {allpenalties3[i]} rss {rss_list[i]} nz {nz_list[i]}")

#Last question 16
final_penalty = allpenalties3[4]
mymodel, nz = read_fit(df=house_data_train_df, l1_penalty=final_penalty)
rss = compute_rss(df=house_data_test_df, model=mymodel)
print(f"final penalty {final_penalty} rss on testdata {rss}")
