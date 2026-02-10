import pandas as pd
import numpy as np

def house_data_dtype_dict():
    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
                'sqft_living15':float, 'grade':int, 'yr_renovated':int,
                'price':float, 'bedrooms':float, 'zipcode':str,
                'long':float, 'sqft_lot15':float, 'sqft_living':float,
                'floors':float, 'condition':int, 'lat':float, 'date':str,
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

def get_numpy_data(df, x_feature_names, y_feature_names):
    H = df[x_feature_names].to_numpy(dtype=float)
    y = df[y_feature_names].to_numpy(dtype=float)
    return H, y

def normalize_features(feature_matrix: np.ndarray):
    squares = np.square(feature_matrix)
    sum_of_squares = np.sum(squares, axis=0)
    Z = np.sqrt(sum_of_squares)
    tiny_threshold = 1e-12
    if len(Z[Z < tiny_threshold]) > 0:
        print("WARNING coefficients close to 0.0 reset to 1.0")
    Z[Z < tiny_threshold] = 1.0
    feature_matrix_norm = feature_matrix / Z
    return feature_matrix_norm, Z

path = r"C:\Users\Evert Jan\courseradatascience\course02\module07\data\\"

x_feature_names = [c for c in list(house_data_dtype_dict().keys()) if c not in ['id', 'date', 'price']]
print(f"Number of X features {len(x_feature_names)}")
y_feature_names = ['price']

house_data_all_df = read_house_data(path=path, file_name="kc_house_data_small.csv", dtype_dict=house_data_dtype_dict())
house_data_train_df = read_house_data(path=path, file_name="kc_house_data_small_train.csv", dtype_dict=house_data_dtype_dict())
house_data_test_df = read_house_data(path=path, file_name="kc_house_data_small_test.csv", dtype_dict=house_data_dtype_dict())
house_data_valid_df = read_house_data(path=path, file_name="kc_house_data_validation.csv", dtype_dict=house_data_dtype_dict())

H, y = get_numpy_data(df=house_data_all_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)
H, y = get_numpy_data(df=house_data_train_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)
H_norm_train, Z_train = normalize_features(feature_matrix=H)
H, y = get_numpy_data(df=house_data_test_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)
H_norm_test = H / Z_train
H, y = get_numpy_data(df=house_data_valid_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)
H_norm_valid = H / Z_train

#Question 7
print("For question 7 : ")
print(H_norm_test[0])
print(H_norm_train[9])
