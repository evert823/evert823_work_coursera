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
    #print("First 5 rows:")
    #print(df.head())

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
    rss_check = np.linalg.norm(error)
    rss = np.sum(error ** 2)
    return rss

def gradient_val(feature_matrix, y, weights):
    y_pred = predict_outcome(feature_matrix=feature_matrix, weights=weights)
    error = y - y_pred
    H_t = feature_matrix.T
    result = np.matmul(H_t, error) * -2
    return result

def gradient_descent_1iter(feature_matrix, y, weights, tolerance, step_size):
    #Calculate the value of the gradient of the RSS for these weights
    G_RSS = gradient_val(feature_matrix=feature_matrix, y=y, weights=weights)

    #Check the magnitude (norm)
    magnitude = np.linalg.norm(G_RSS)

    converged = magnitude < tolerance

    #Update the weights
    new_weights = weights.copy()
    if converged == False:
        #for i in range(weights.shape[0]):
        #    new_weights[i][0] -= G_RSS[i][0] * step_size
        new_weights = new_weights - step_size * G_RSS
    
    return new_weights, converged, magnitude, G_RSS

def gradient_descent(feature_matrix, y, init_weights, tolerance, step_size, max_iter):
    itercount = 0
    new_weights, converged, magnitude, G_RSS = gradient_descent_1iter(feature_matrix=feature_matrix,
                                                    y=y,
                                                    weights=init_weights,
                                                    tolerance=tolerance,
                                                    step_size=step_size)
    
    while converged == False and (itercount < max_iter or max_iter == -1):
        itercount += 1
        if itercount % 10000 == 0:
            print (f"new weights during gradient descent {new_weights.T} magnitude {magnitude} gradient of RSS {G_RSS.T}")
        new_weights, converged, magnitude, G_RSS = gradient_descent_1iter(feature_matrix=feature_matrix,
                                                        y=y,
                                                        weights=new_weights,
                                                        tolerance=tolerance,
                                                        step_size=step_size)

    print(f"new_weights after gradient descent {new_weights.T} itercount {itercount}")

    return new_weights


np.set_printoptions(suppress=True, formatter={'float_kind': '{:f}'.format}, linewidth=120)

path = r"C:\Users\Evert Jan\courseradatascience\course02\module03\data\\"
file_name_inp = "kc_house_train_data.csv"
#file_name_inp = "mockdata.csv"

#x_feature_names = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
#x_feature_names = ['sqft_living']
x_feature_names = ['sqft_living', 'sqft_living15']
y_feature_names = ['price']

house_data_df = read_house_data(path=path, file_name=file_name_inp, dtype_dict=house_data_dtype_dict())
H, y = get_numpy_data(df=house_data_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)

init_weights = np.array([[-100000.0], [1.0], [1.0]])
new_weights = gradient_descent(feature_matrix=H,
                                y=y,
                                init_weights=init_weights,
                                tolerance=1e9,
                                step_size=4e-12,
                                max_iter=100000)

file_name_test = "kc_house_test_data.csv"
house_test_data_df = read_house_data(path=path, file_name=file_name_test, dtype_dict=house_data_dtype_dict())
H_test, y_test = get_numpy_data(df=house_test_data_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)

y_test_pred = predict_outcome(feature_matrix=H_test, weights=new_weights)
print(y_test_pred)

rss = RSS(feature_matrix=H_test,y=y_test,weights=new_weights)
print(f"RSS on testdata {rss}")
