import pandas as pd

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

def simple_linear_regression(input_feature, output):
    #GPT-5 mini created this based on LaTeX file
    # convert to 1D numpy arrays
    x = input_feature.values.squeeze()
    y = output.values.squeeze()

    N = x.shape[0]
    A = y.sum()
    B = x.sum()
    C = (x * y).sum()
    D = (x * x).sum()

    denom = D - (B * B) / N
    if denom == 0:
        raise ValueError("Division by zero in slope computation")

    w1 = (C - (A * B) / N) / denom  # slope
    w0 = A / N - w1 * (B / N)       # intercept

    return (float(w0), float(w1))

def get_regression_predictions(input_feature, intercept, slope):
    x = input_feature.values.squeeze()
    preds = intercept + slope * x
    preds_df = pd.DataFrame(preds, index=input_feature.index, columns=['prediction'])
    return preds_df

def inverse_regression_predictions(output, intercept, slope):
    y = output.values.squeeze()
    x_pred = (y - intercept) / slope
    x_pred_df = pd.DataFrame(x_pred, index=output.index, columns=['predicted_sqft'])
    return x_pred_df

def question_06(one_x, intercept, slope):
    one_price = intercept + slope * one_x
    print(f"Answer to question 6 price : {one_price}")

def question_10(one_y, intercept, slope):
    one_x = (one_y - intercept) / slope
    print(f"Answer to question 10 sqft : {one_x}")

def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    preds_df = get_regression_predictions(input_feature=input_feature, intercept=intercept, slope=slope)
    y = output.values.squeeze()
    y_pred = preds_df.values.squeeze()
    rs = (y - y_pred) * (y - y_pred)
    rss = rs.sum()
    return(rss)

path = r"C:\Users\Evert Jan\courseradatascience\course02\module02\data\\"
file_name_inp = "kc_house_train_data.csv"
#file_name_inp = "mockdata.csv"

house_data_df = read_house_data(path=path, file_name=file_name_inp, dtype_dict=house_data_dtype_dict())
assess_dataframe(house_data_df)

input_feature_columns = ['sqft_living']
#input_feature_columns = ['bedrooms']

input_feature = house_data_df[input_feature_columns]
output = house_data_df[['price']]
w = simple_linear_regression(input_feature=input_feature, output=output)
print(f"Learned parameters : {w}")

question_06(one_x=2650, intercept=w[0], slope=w[1])

rss = get_residual_sum_of_squares(input_feature=input_feature, output=output, intercept=w[0], slope=w[1])
print(f"RSS on the training data : {rss}")

x_pred_df = inverse_regression_predictions(output=output, intercept=w[0], slope=w[1])
assess_dataframe(x_pred_df)

question_10(one_y=800000, intercept=w[0], slope=w[1])

#Now we load the test data but re-use the intercept and slope from train data
file_name_inp = "kc_house_test_data.csv"
house_data_df = read_house_data(path=path, file_name=file_name_inp, dtype_dict=house_data_dtype_dict())
input_feature = house_data_df[input_feature_columns]
output = house_data_df[['price']]
rss = get_residual_sum_of_squares(input_feature=input_feature, output=output, intercept=w[0], slope=w[1])
print(f"RSS on the test data : {rss}")
