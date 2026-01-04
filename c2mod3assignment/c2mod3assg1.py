import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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

def add_features(df):
    df_out = df.copy()
    df_out['bedrooms_squared'] = df_out['bedrooms'] ** 2
    df_out['bed_bath_rooms'] = df_out['bedrooms'] * df_out['bathrooms']
    df_out['log_sqft_living'] = np.log(df_out['sqft_living'])
    df_out['lat_plus_long'] = df_out['lat'] + df_out['long']
    return df_out

def question04(df):
    cols = ['bedrooms_squared', 'bed_bath_rooms', 'log_sqft_living', 'lat_plus_long']
    means = np.mean(df[cols].values, axis=0)
    a = zip(cols, means)
    b = dict(a)
    print(f"question 4 : {b}")


path = r"C:\Users\Evert Jan\courseradatascience\course02\module03\data\\"
file_name_inp = "kc_house_train_data.csv"
#file_name_inp = "kc_house_test_data.csv"
#file_name_inp = "mockdata.csv"
house_data_df = read_house_data(path=path, file_name=file_name_inp, dtype_dict=house_data_dtype_dict())
house_data_2_df = add_features(df=house_data_df)
#assess_dataframe(house_data_2_df)
question04(house_data_2_df)


feature_names_model1 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
feature_names_model2 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long',
                        'bed_bath_rooms']
feature_names_model3 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long',
                        'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']


#Here do the linear regression with sklearn model1
y = house_data_2_df['price'].values
X_df = house_data_2_df[feature_names_model1]
X = X_df.values
mylinreg = LinearRegression()
mylinreg.fit(X, y)
coeffs = dict(zip(feature_names_model1, mylinreg.coef_))
coeffs['intercept'] = mylinreg.intercept_
print(f"par + intercept model1 : {coeffs}")
temp_df = house_data_2_df.copy()
temp_df['predicted_price'] = mylinreg.predict(X=X)
temp_df['rs'] = (temp_df['price'] - temp_df['predicted_price']) * (temp_df['price'] - temp_df['predicted_price'])
rss = np.sum(temp_df['rs'].values, axis=0)
print(f"rss model1 {rss}\n")

house_test_data_df = read_house_data(path=path, file_name='kc_house_test_data.csv', dtype_dict=house_data_dtype_dict())
house_test_data_2_df = add_features(df=house_test_data_df)
X_df = house_test_data_2_df[feature_names_model1]
X = X_df.values
temp_df = house_test_data_2_df.copy()
temp_df['predicted_price'] = mylinreg.predict(X=X)
temp_df['rs'] = (temp_df['price'] - temp_df['predicted_price']) * (temp_df['price'] - temp_df['predicted_price'])
rss = np.sum(temp_df['rs'].values, axis=0)
print(f"rss model1 test_data {rss}\n")

#Here do the linear regression with sklearn model2
y = house_data_2_df['price'].values
X_df = house_data_2_df[feature_names_model2]
X = X_df.values
mylinreg = LinearRegression()
mylinreg.fit(X, y)
coeffs = dict(zip(feature_names_model2, mylinreg.coef_))
coeffs['intercept'] = mylinreg.intercept_
print(f"par + intercept model2 : {coeffs}")
temp_df = house_data_2_df.copy()
temp_df['predicted_price'] = mylinreg.predict(X=X)
temp_df['rs'] = (temp_df['price'] - temp_df['predicted_price']) * (temp_df['price'] - temp_df['predicted_price'])
rss = np.sum(temp_df['rs'].values, axis=0)
print(f"rss model2 {rss}\n")

house_test_data_df = read_house_data(path=path, file_name='kc_house_test_data.csv', dtype_dict=house_data_dtype_dict())
house_test_data_2_df = add_features(df=house_test_data_df)
X_df = house_test_data_2_df[feature_names_model2]
X = X_df.values
temp_df = house_test_data_2_df.copy()
temp_df['predicted_price'] = mylinreg.predict(X=X)
temp_df['rs'] = (temp_df['price'] - temp_df['predicted_price']) * (temp_df['price'] - temp_df['predicted_price'])
rss = np.sum(temp_df['rs'].values, axis=0)
print(f"rss model2 test_data {rss}\n")

#Here do the linear regression with sklearn model3
y = house_data_2_df['price'].values
X_df = house_data_2_df[feature_names_model3]
X = X_df.values
mylinreg = LinearRegression()
mylinreg.fit(X, y)
coeffs = dict(zip(feature_names_model3, mylinreg.coef_))
coeffs['intercept'] = mylinreg.intercept_
print(f"par + intercept model3 : {coeffs}")
temp_df = house_data_2_df.copy()
temp_df['predicted_price'] = mylinreg.predict(X=X)
temp_df['rs'] = (temp_df['price'] - temp_df['predicted_price']) * (temp_df['price'] - temp_df['predicted_price'])
rss = np.sum(temp_df['rs'].values, axis=0)
print(f"rss model3 {rss}\n")

house_test_data_df = read_house_data(path=path, file_name='kc_house_test_data.csv', dtype_dict=house_data_dtype_dict())
house_test_data_2_df = add_features(df=house_test_data_df)
X_df = house_test_data_2_df[feature_names_model3]
X = X_df.values
temp_df = house_test_data_2_df.copy()
temp_df['predicted_price'] = mylinreg.predict(X=X)
temp_df['rs'] = (temp_df['price'] - temp_df['predicted_price']) * (temp_df['price'] - temp_df['predicted_price'])
rss = np.sum(temp_df['rs'].values, axis=0)
print(f"rss model3 test_data {rss}\n")
