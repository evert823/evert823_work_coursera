import pandas as pd
import numpy as np

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

def get_numpy_data(df, x_feature_names, y_feature_names):
    H = df[x_feature_names].to_numpy(dtype=float)
    ones = np.ones((H.shape[0], 1), dtype=float)
    H = np.hstack((ones, H)) #The coeff mapped to this extra constant feature is the intercept
    y = df[y_feature_names].to_numpy(dtype=float)
    return H, y

def predict_outcome(feature_matrix, weights):
    y_pred = np.matmul(feature_matrix, weights)
    return(y_pred)

def normalize_features(feature_matrix: np.ndarray):
    squares = np.square(feature_matrix)
    sum_of_squares = np.sum(squares, axis=0)
    Z = np.sqrt(sum_of_squares)
    feature_matrix_norm = feature_matrix / Z
    return feature_matrix_norm, Z

def lasso_rho_normalized_input(j: int,
                              feature_matrix_norm:np.ndarray,
                              y:np.ndarray,
                              weights: np.ndarray):
    #weightsxj = weights without j-th coefficient
    weights_xj = weights.copy()
    weights_xj[j, 0] = 0

    y_pred_xj = predict_outcome(feature_matrix=feature_matrix_norm, weights=weights_xj)
    h_j = feature_matrix_norm[:, j].reshape(-1, 1)
    result1 = np.matmul(h_j.T, y - y_pred_xj)
    return result1[0, 0]


path = r"C:\Users\Evert Jan\courseradatascience\course02\module06\data2\\"

house_data_all_df = read_house_data(path=path, file_name="kc_house_data.csv", dtype_dict=house_data_dtype_dict())
#house_data_train_df = read_house_data(path=path, file_name="kc_house_train_data.csv", dtype_dict=house_data_dtype_dict())
#house_data_test_df = read_house_data(path=path, file_name="kc_house_test_data.csv", dtype_dict=house_data_dtype_dict())

x_feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot']
y_feature_names = ['price']
H, y = get_numpy_data(df=house_data_all_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)

init_weights=np.array([[10.0], [1.0], [2.0], [1.0], [2.0]])
H_norm, Z = normalize_features(feature_matrix=H)
myrho = lasso_rho_normalized_input(j = 1,
                                   feature_matrix_norm=H_norm,
                                   y=y,
                                   weights=init_weights)
