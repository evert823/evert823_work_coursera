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
                       normalize_dummy_ones: bool):
    squares = np.square(feature_matrix)
    sum_of_squares = np.sum(squares, axis=0)
    Z = np.sqrt(sum_of_squares)
    if normalize_dummy_ones == False:
        Z[0] = 1.0
    tiny_threshold = 1e-12
    Z[Z < tiny_threshold] = 1.0
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

def do_coordinate_descent_lasso_norm(init_weights: np.ndarray,
                                     l1_penalty: float,
                                     feature_matrix: np.ndarray,
                                     y: np.ndarray,
                                     epsilon: float,
                                     max_iterations: int,
                                     normalize_dummy_ones: bool,
                                     print_init_rho=False):
    '''
    This implementation first normalizes the features to L2 norm
    This implementation expects the feature matrix as numpy array
    This implementation expects a dummy ones column
    that can be mapped to the increment as 0th coefficient

    '''

    if not isinstance(feature_matrix, np.ndarray):
        raise TypeError("feature_matrix must be a numpy.ndarray")
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be 2-dimensional")
    if feature_matrix.shape[0] == 0:
        raise ValueError("feature_matrix must have at least one row")
    first_col = feature_matrix[:, 0]
    if not np.allclose(first_col, 1.0, rtol=1e-8, atol=1e-12):
        raise ValueError("Expected first column to be a dummy column of ones")

    H_norm, Z = normalize_features(feature_matrix=feature_matrix,
                                   normalize_dummy_ones=normalize_dummy_ones)

    max_update_step = epsilon + 1
    done_iterations = 0
    D = H_norm.shape[1]
    new_weights = init_weights.copy()

    while max_update_step > epsilon and done_iterations < max_iterations:
        max_update_step = 0
        for j in range(D):
            rho_j = lasso_rho_normalized_input(j=j,
                                               feature_matrix_norm=H_norm,
                                               y=y,
                                               weights=new_weights)
            if done_iterations < 1 and print_init_rho == True:
                print(f"j {j} rho_j {rho_j}")
            w_prev = new_weights[j, 0]
            if j == 0:
                new_weights[j, 0] = rho_j
            elif rho_j < -1 * l1_penalty / 2:
                new_weights[j, 0] = rho_j + (l1_penalty / 2)
            elif rho_j > l1_penalty / 2:
                new_weights[j, 0] = rho_j - (l1_penalty / 2)
            else:
                new_weights[j, 0] = 0.0
            update_step = np.abs(w_prev - new_weights[j, 0])
            if update_step > max_update_step:
                max_update_step = update_step
        done_iterations += 1

    print(f"done_iterations {done_iterations}")
    return new_weights, Z



path = r"C:\Users\Evert Jan\courseradatascience\course02\module06\data2\\"

house_data_all_df = read_house_data(path=path, file_name="kc_house_data.csv", dtype_dict=house_data_dtype_dict())
house_data_train_df = read_house_data(path=path, file_name="kc_house_train_data.csv", dtype_dict=house_data_dtype_dict())
house_data_test_df = read_house_data(path=path, file_name="kc_house_test_data.csv", dtype_dict=house_data_dtype_dict())
house_data_all_df = fix_data(df=house_data_all_df)
house_data_train_df = fix_data(df=house_data_train_df)
house_data_test_df = fix_data(df=house_data_test_df)

x_feature_names = ['sqft_living', 'bedrooms']
y_feature_names = ['price']
H, y = get_numpy_data(df=house_data_all_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)

print("Starting with question 10")

init_weights = np.array([[1.0], [4.0], [1.0]])
l1_penalty = 1e7
max_iterations = 1000
epsilon = 1.0

new_weights, Z = do_coordinate_descent_lasso_norm(init_weights=init_weights,
                                                  l1_penalty=l1_penalty,
                                                  feature_matrix=H,
                                                  y=y,
                                                  epsilon=epsilon,
                                                  max_iterations=max_iterations,
                                                  normalize_dummy_ones=True,
                                                  print_init_rho=True)
print(f"new_weights \n{new_weights}\nZ {Z}")

#For question 15:
print("Starting with question 15")
init_weights = np.zeros((H.shape[1], 1), dtype=float)
l1_penalty = 1e7
max_iterations = 1000
epsilon = 1.0

new_weights, Z = do_coordinate_descent_lasso_norm(init_weights=init_weights,
                                                  l1_penalty=l1_penalty,
                                                  feature_matrix=H,
                                                  y=y,
                                                  epsilon=epsilon,
                                                  max_iterations=max_iterations,
                                                  normalize_dummy_ones=True)
print(f"new_weights \n{new_weights}\nZ {Z}")

H_norm, Z = normalize_features(feature_matrix=H,
                               normalize_dummy_ones=True)
rss = RSS(feature_matrix=H_norm, y=y, weights=new_weights)
print (f"rss question 15 {rss}")

#Question 17 and further
print("Starting with question 17 and further")

x_feature_names_2 = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
                     'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
H, y = get_numpy_data(df=house_data_train_df, x_feature_names=x_feature_names_2, y_feature_names=y_feature_names)

init_weights = np.zeros((H.shape[1], 1), dtype=float)
max_iterations = 1000
epsilon = 1.0

l1_penalty = 1e7
weights_1e7, Z_1e7 = do_coordinate_descent_lasso_norm(init_weights=init_weights,
                                                  l1_penalty=l1_penalty,
                                                  feature_matrix=H,
                                                  y=y,
                                                  epsilon=epsilon,
                                                  max_iterations=max_iterations,
                                                  normalize_dummy_ones=True)
print(f"weights_1e7.T \n{weights_1e7.T}\nZ {Z_1e7}")

l1_penalty = 1e8
weights_1e8, Z_1e8 = do_coordinate_descent_lasso_norm(init_weights=init_weights,
                                                  l1_penalty=l1_penalty,
                                                  feature_matrix=H,
                                                  y=y,
                                                  epsilon=epsilon,
                                                  max_iterations=max_iterations,
                                                  normalize_dummy_ones=True)
print(f"weights_1e8.T \n{weights_1e8.T}\nZ {Z_1e8}")

l1_penalty = 1e4
epsilon = 5e5
weights_1e4, Z_1e4 = do_coordinate_descent_lasso_norm(init_weights=init_weights,
                                                  l1_penalty=l1_penalty,
                                                  feature_matrix=H,
                                                  y=y,
                                                  epsilon=epsilon,
                                                  max_iterations=max_iterations,
                                                  normalize_dummy_ones=True)
print(f"weights_1e4.T \n{weights_1e4.T}\nZ {Z_1e4}")

#We can rescale weights so that we can use them for making predictions on testdata, if
# we don't want to normalize the testdata
#The impact of normalizing was that we divided Xj by Zj while keeping y the same
# so that wj INCREASED by factor Zj
#So for this purpose we must DIVIDE the weights by the scaling factor Z
rescaled_weights_1e7 = weights_1e7 / Z_1e7.reshape(-1, 1)
rescaled_weights_1e8 = weights_1e8 / Z_1e8.reshape(-1, 1)
rescaled_weights_1e4 = weights_1e4 / Z_1e4.reshape(-1, 1)

print(f"check question 26 rescaled_weights_1e7[3] {rescaled_weights_1e7[3]}")

#house_data_test_df
H, y = get_numpy_data(df=house_data_test_df, x_feature_names=x_feature_names_2, y_feature_names=y_feature_names)
rss_1e7 = RSS(feature_matrix=H, y=y, weights=rescaled_weights_1e7)
rss_1e8 = RSS(feature_matrix=H, y=y, weights=rescaled_weights_1e8)
rss_1e4 = RSS(feature_matrix=H, y=y, weights=rescaled_weights_1e4)
print(f"rss_1e7 {rss_1e7} rss_1e8 {rss_1e8} rss_1e4 {rss_1e4}")
