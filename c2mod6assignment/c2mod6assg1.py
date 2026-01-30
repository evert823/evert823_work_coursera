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

def normalize_features(df, feature_names: list[str], mean_center=False):
    df2 = df.copy()
    for fn in feature_names:

        #GHCP suggests to mean-center the data before doing the L2 scaling
        if mean_center == False:
            cols = df[fn]
        else:
            mymean = df[fn].mean()
            cols = df[fn] - mymean
        sum_squares = (cols ** 2).sum()
        Zj = sqrt(sum_squares)
        df2[fn] = cols / Zj
    return df2

def mean_center_features(df, feature_names: list[str]):
    df2 = df.copy()
    for fn in feature_names:
        mymean = df[fn].mean()
        df2[fn] = df[fn] - mymean
    return df2


path = r"C:\Users\Evert Jan\courseradatascience\course02\module06\data\\"

house_data_all_df = read_house_data(path=path, file_name="kc_house_data.csv", dtype_dict=house_data_dtype_dict())
house_data_all_df = add_columns(df=house_data_all_df)
assess_dataframe(df=house_data_all_df)

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

#house_data_all_df = normalize_features(df=house_data_all_df, feature_names=all_features, mean_center=True)
#house_data_all_df = mean_center_features(df=house_data_all_df, feature_names=['price'])
#assess_dataframe(df=house_data_all_df)

mymodel = linear_model.Lasso(alpha=5e2)
mymodel.fit(house_data_all_df[all_features], house_data_all_df['price'])
a = np.array([mymodel.intercept_])
w = np.hstack((a, mymodel.coef_))
print(w)
