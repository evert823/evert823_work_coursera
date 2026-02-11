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

def distances(feature_matrix: np.ndarray, input_observation: np.ndarray):
    diff = feature_matrix - input_observation
    squares = np.square(diff)
    sum_of_squares = np.sum(squares, axis=1)
    result = np.sqrt(sum_of_squares)
    return result

def k_nearest_datapoints(feature_matrix: np.ndarray,
                         input_observation: np.ndarray,
                         k: int):
    '''
    Find the k nearest datapoints from feature_matrix
    given input_observation
    return a list of tuples (index in feature_matrix, distance)
    sorted by distance asc
    '''
    N = feature_matrix.shape[0]
    assert k > 0
    assert k <= N
    mydistances = distances(feature_matrix=feature_matrix,
                            input_observation=input_observation)
    
    '''
    In queue we will store tuples of
    - index i in feature_matrix
    - distance from row[i] in feature_matrix to input_observation
    '''
    queue: list[tuple[int, float]] = [(i, mydistances[i]) for i in range(k)]
    queue.sort(key=lambda x: x[1])

    for i in range(k,N):
        d = mydistances[i]
        j = k
        while j > 0 and queue[j - 1][1] > d:
            j = j - 1
        if j < k:
            queue.insert(j, (i, d))
            queue.pop()
    
    return queue

def predict_knn(feature_matrix: np.ndarray,
                y: np.ndarray,
                input_observation: np.ndarray,
                k: int):
    try:
        kn_queue = k_nearest_datapoints(feature_matrix=feature_matrix,
                                        input_observation=input_observation,
                                        k=k)
    except:
        print("Not able to find k nearest neighbours")
    
    total_price: float = 0.0
    for j in range(k):
        i = kn_queue[j][0]
        price = y[i][0]
        total_price += price
    return total_price / k

path = r"C:\Users\Evert Jan\courseradatascience\course02\module07\data\\"

x_feature_names = [c for c in list(house_data_dtype_dict().keys()) if c not in ['id', 'date', 'price']]
print(f"Number of X features {len(x_feature_names)}")
y_feature_names = ['price']

house_data_all_df = read_house_data(path=path, file_name="kc_house_data_small.csv", dtype_dict=house_data_dtype_dict())
house_data_train_df = read_house_data(path=path, file_name="kc_house_data_small_train.csv", dtype_dict=house_data_dtype_dict())
house_data_test_df = read_house_data(path=path, file_name="kc_house_data_small_test.csv", dtype_dict=house_data_dtype_dict())
house_data_valid_df = read_house_data(path=path, file_name="kc_house_data_validation.csv", dtype_dict=house_data_dtype_dict())

H, y_all = get_numpy_data(df=house_data_all_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)
H, y_train = get_numpy_data(df=house_data_train_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)
H_norm_train, Z_train = normalize_features(feature_matrix=H)
H, y_test = get_numpy_data(df=house_data_test_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)
H_norm_test = H / Z_train
H, y_valid = get_numpy_data(df=house_data_valid_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)
H_norm_valid = H / Z_train

#Question 7
print("For question 7 : ")
print(H_norm_train[9])
print(H_norm_test[0])
a = np.linalg.norm(H_norm_train[9] - H_norm_test[0])
print(a)

b = distances(feature_matrix=H_norm_train, input_observation=H_norm_test[0])
print(b[9])

#Question 10
print("For question 10 : ")
for i in range(10):
    print(b[i])

#Question 14
print("For question 14 : ")
print(b[100])


#Question 16
print("For question 16 : ")
k = 4
i = 2
kn_queue = k_nearest_datapoints(feature_matrix=H_norm_train,
                                input_observation=H_norm_test[i],
                                k=k)
print(f"k {k} testrow {i} kn_queue {kn_queue}")

#Question 17
print("For question 17 : ")
i = kn_queue[0][0]
print(i)
print(house_data_train_df.iloc[i])

#Question 21
print("For question 21 : ")
k = 4
i = 2
predicted_price = predict_knn(feature_matrix=H_norm_train,
                              y=y_train,
                              input_observation=H_norm_test[i],
                              k=k)
print(predicted_price)

