import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def gradient_val(feature_matrix, y, weights, l2_penalty):
    y_pred = predict_outcome(feature_matrix=feature_matrix, weights=weights)
    error = y - y_pred
    H_t = feature_matrix.T
    result = np.matmul(H_t, error) * -2
    #result.shape == (2,1)
    for j in range(result.shape[0]):
        if j > 0:
            result[j][0] += 2 * l2_penalty * weights[j][0]
    return result

def gradient_descent_1iter(feature_matrix, y, weights, tolerance, step_size, l2_penalty):
    #Calculate the value of the gradient of the RSS for these weights
    G_RSS = gradient_val(feature_matrix=feature_matrix, y=y, weights=weights, l2_penalty=l2_penalty)

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

def gradient_descent(feature_matrix, y, init_weights, tolerance, step_size, max_iter, l2_penalty):
    itercount = 0
    new_weights, converged, magnitude, G_RSS = gradient_descent_1iter(feature_matrix=feature_matrix,
                                                    y=y,
                                                    weights=init_weights,
                                                    tolerance=tolerance,
                                                    step_size=step_size,
                                                    l2_penalty=l2_penalty)
    
    while converged == False and (itercount < max_iter or max_iter == -1):
        itercount += 1
        if itercount % 1000000 == 0:
            print (f"new weights during gradient descent {new_weights.T} magnitude {magnitude} gradient of RSS {G_RSS.T}")
        new_weights, converged, magnitude, G_RSS = gradient_descent_1iter(feature_matrix=feature_matrix,
                                                        y=y,
                                                        weights=new_weights,
                                                        tolerance=tolerance,
                                                        step_size=step_size,
                                                        l2_penalty=l2_penalty)

    print(f"new_weights after gradient descent {new_weights.T} itercount {itercount}")

    return new_weights



np.set_printoptions(suppress=True, formatter={'float_kind': '{:f}'.format}, linewidth=120)

path = r"C:\Users\Evert Jan\courseradatascience\course02\module05\data\\"

mockdata_df = read_house_data(path=path, file_name="mockdata.csv", dtype_dict=house_data_dtype_dict())
assess_dataframe(mockdata_df)

x_feature_names = ['sqft_living']
y_feature_names = ['price']
H, y = get_numpy_data(df=mockdata_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)

init_weights = np.array([[99999.0], [99.0]])
print(f"Start processing mockdata")
new_weights = gradient_descent(feature_matrix=H,
                                y=y,
                                init_weights=init_weights,
                                tolerance=1e-1,
                                step_size=3e-10,
                                max_iter=100000,
                                l2_penalty=0.0)


house_data_train_df = read_house_data(path=path, file_name="kc_house_train_data.csv", dtype_dict=house_data_dtype_dict())
house_data_train_df = house_data_train_df.sort_values(by=['sqft_living', 'price']).reset_index(drop=True)

assess_dataframe(house_data_train_df)
x_feature_names = ['sqft_living']
y_feature_names = ['price']
H, y = get_numpy_data(df=house_data_train_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)

init_weights = np.array([[0.0], [0.0]])
print(f"Start processing training data")
simple_weights_0_penalty = gradient_descent(feature_matrix=H,
                                y=y,
                                init_weights=init_weights,
                                tolerance=1e-1,
                                step_size=1e-12,
                                max_iter=1000,
                                l2_penalty=0.0)

init_weights = np.array([[0.0], [0.0]])
print(f"Start processing training data and high l2_penalty")
simple_weights_high_penalty = gradient_descent(feature_matrix=H,
                                y=y,
                                init_weights=init_weights,
                                tolerance=1e-1,
                                step_size=1e-12,
                                max_iter=1000,
                                l2_penalty=1e11)

y_0p = predict_outcome(H, simple_weights_0_penalty)
y_hp = predict_outcome(H, simple_weights_high_penalty)
plt.figure(figsize=(20, 12), dpi=120)
plt.plot(H,y,'k.', H,y_0p,'b-', H,y_hp,'r-')
plt.savefig('.\\images\\assignment2_14.png')


house_data_test_df = read_house_data(path=path, file_name="kc_house_test_data.csv", dtype_dict=house_data_dtype_dict())
house_data_test_df = house_data_test_df.sort_values(by=['sqft_living', 'price']).reset_index(drop=True)
assess_dataframe(house_data_test_df)
H_test, y_test = get_numpy_data(df=house_data_test_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)
rss_test_0w = RSS(feature_matrix=H_test, y=y_test, weights=init_weights)
rss_test_0p = RSS(feature_matrix=H_test, y=y_test, weights=simple_weights_0_penalty)
rss_test_hp = RSS(feature_matrix=H_test, y=y_test, weights=simple_weights_high_penalty)
print(f"rss_test_0w {rss_test_0w} rss_test_0p {rss_test_0p} rss_test_hp {rss_test_hp}")

#For question 19 we extend the list of features
x_feature_names = ['sqft_living', 'sqft_living15']
y_feature_names = ['price']
#then rebuild the numpy matrices
H, y = get_numpy_data(df=house_data_train_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)
H_test, y_test = get_numpy_data(df=house_data_test_df, x_feature_names=x_feature_names, y_feature_names=y_feature_names)

init_weights = np.array([[0.0], [0.0], [0.0]])
multiple_weights_0_penalty = gradient_descent(feature_matrix=H,
                                y=y,
                                init_weights=init_weights,
                                tolerance=1e-1,
                                step_size=1e-12,
                                max_iter=1000,
                                l2_penalty=0.0)

init_weights = np.array([[0.0], [0.0], [0.0]])
multiple_weights_high_penalty = gradient_descent(feature_matrix=H,
                                y=y,
                                init_weights=init_weights,
                                tolerance=1e-1,
                                step_size=1e-12,
                                max_iter=1000,
                                l2_penalty=1e11)

print(f"multiple_weights_0_penalty \n {multiple_weights_0_penalty}")
print(f"multiple_weights_high_penalty \n {multiple_weights_high_penalty}")

rss_test_0w = RSS(feature_matrix=H_test, y=y_test, weights=init_weights)
rss_test_0p = RSS(feature_matrix=H_test, y=y_test, weights=multiple_weights_0_penalty)
rss_test_hp = RSS(feature_matrix=H_test, y=y_test, weights=multiple_weights_high_penalty)
print(f"rss_test_0w {rss_test_0w} rss_test_0p {rss_test_0p} rss_test_hp {rss_test_hp}")
