import pandas as pd
import numpy as np
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

def fix_data(df):
    df2 = df.copy()
    df2['floors'] = df2['floors'].astype(float)
    return df2

def assess_dataframe(df):
    print(f"rowcount {df.shape[0]} colcount {df.shape[1]}")
    print(f"dtypes {dict(df.dtypes)}")
    print("First 5 rows:")
    print(df.head())

def get_numpy_data(df, x_feature_names, y_feature_names):
    H = df[x_feature_names].to_numpy(dtype=float)
    ones = np.ones((H.shape[0], 1), dtype=float)
    H = np.hstack((ones, H)) #The coeff mapped to this extra constant feature is the intercept
    y = df[y_feature_names].to_numpy(dtype=float)
    return H, y

def predict_outcome(feature_matrix, weights):
    y_pred = np.matmul(feature_matrix, weights)
    return(y_pred)

def RSS(feature_matrix, y, weights):
    y_pred = predict_outcome(feature_matrix=feature_matrix, weights=weights)
    error = y - y_pred
    rss = np.sum(error ** 2)
    return rss

def normalize_features(feature_matrix: np.ndarray,
                       normalize_dummy_ones=True,
                       do_normalize=False):
    squares = np.square(feature_matrix)
    sum_of_squares = np.sum(squares, axis=0)

    if do_normalize == True:
        Z = np.sqrt(sum_of_squares)
    else:
        Z = np.ones((H.shape[1],), dtype=float)

    if normalize_dummy_ones == False:
        Z[0] = 1.0

    feature_matrix_norm = feature_matrix / Z
    return feature_matrix_norm, Z

path = r"C:\Users\Evert Jan\courseradatascience\course02\module06\data2\\"

house_data_mock_df = read_house_data(path=path, file_name="mockdata.csv", dtype_dict=house_data_dtype_dict())

x_feature_names = ['sqft_living']
y_feature_names = ['price']
H, y = get_numpy_data(df=house_data_mock_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)

print("\nwithout normalization")
H_norm, Z = normalize_features(feature_matrix=H,
                               normalize_dummy_ones=True,
                               do_normalize=False)
print(f"H_norm {H_norm}")
print(f"Z {Z}")
mymodel = linear_model.LinearRegression()
mymodel.fit(H_norm, y)
a = np.array([mymodel.intercept_])
w = np.hstack((a, mymodel.coef_))
print(w)

print("\nWITH normalization")
H_norm, Z = normalize_features(feature_matrix=H,
                               normalize_dummy_ones=True,
                               do_normalize=True)
print(f"H_norm {H_norm}")
print(f"Z {Z}")
mymodel = linear_model.LinearRegression()
mymodel.fit(H_norm, y)
a = np.array([mymodel.intercept_])
w = np.hstack((a, mymodel.coef_))
print(w)
