import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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
    print("First 5 rows:")
    print(df.head())

def add_terms_of_polynomial(df, feature_name, degree):
    #Add columns polynomial terms as columns to dataframe till/including degree
    df_out = df.copy()

    for i in range(2, degree + 1):
        fn = f"{feature_name}_pwr_{i}"
        df_out[fn] = df_out[feature_name] ** i

    return df_out


path = r"C:\Users\Evert Jan\courseradatascience\course02\module04\data\\"
file_name_mock = "mockdata.csv"
file_name_all = "kc_house_data.csv"
file_name_train = "wk3_kc_house_train_data.csv"
file_name_test = "wk3_kc_house_test_data.csv"
file_name_valid = "wk3_kc_house_valid_data.csv"
file_name_set_1 = "wk3_kc_house_set_1_data.csv"
file_name_set_2 = "wk3_kc_house_set_2_data.csv"
file_name_set_3 = "wk3_kc_house_set_3_data.csv"
file_name_set_4 = "wk3_kc_house_set_4_data.csv"
house_data_mock_df = read_house_data(path=path, file_name=file_name_mock, dtype_dict=house_data_dtype_dict())
house_data_mock_df.sort_values(['sqft_living', 'price'])
house_data_all_df = read_house_data(path=path, file_name=file_name_all, dtype_dict=house_data_dtype_dict())
house_data_all_df.sort_values(['sqft_living', 'price'])
house_data_train_df = read_house_data(path=path, file_name=file_name_train, dtype_dict=house_data_dtype_dict())
house_data_train_df.sort_values(['sqft_living', 'price'])
house_data_test_df = read_house_data(path=path, file_name=file_name_test, dtype_dict=house_data_dtype_dict())
house_data_test_df.sort_values(['sqft_living', 'price'])
house_data_valid_df = read_house_data(path=path, file_name=file_name_valid, dtype_dict=house_data_dtype_dict())
house_data_valid_df.sort_values(['sqft_living', 'price'])
house_data_set_1_df = read_house_data(path=path, file_name=file_name_set_1, dtype_dict=house_data_dtype_dict())
house_data_set_1_df.sort_values(['sqft_living', 'price'])
house_data_set_2_df = read_house_data(path=path, file_name=file_name_set_2, dtype_dict=house_data_dtype_dict())
house_data_set_2_df.sort_values(['sqft_living', 'price'])
house_data_set_3_df = read_house_data(path=path, file_name=file_name_set_3, dtype_dict=house_data_dtype_dict())
house_data_set_3_df.sort_values(['sqft_living', 'price'])
house_data_set_4_df = read_house_data(path=path, file_name=file_name_set_4, dtype_dict=house_data_dtype_dict())
house_data_set_4_df.sort_values(['sqft_living', 'price'])

#define degree, x_feature_names and y_feature_names
degree = 1
x_feature_names = ['sqft_living']
for i in range(2, degree + 1):
    x_feature_names.append(f'sqft_living_pwr_{i}')
y_feature_names = ['price']

#Create the dataframe with terms of polynomial of given degree
polynomial_df = add_terms_of_polynomial(df=house_data_all_df, feature_name='sqft_living', degree=degree)

#Create and fit a model
y = polynomial_df[y_feature_names].values
X = polynomial_df[x_feature_names].values
mylinreg = LinearRegression()
mylinreg.fit(X, y)

w = np.hstack((mylinreg.intercept_.reshape(1, 1), mylinreg.coef_))
print(f"w : {w} shape : {w.shape}")

y_pred = mylinreg.predict(X)
plt.figure(figsize=(16, 9), dpi=100)
plt.plot(polynomial_df['sqft_living'],polynomial_df['price'],'.',
         polynomial_df['sqft_living'], y_pred,'-')
plt.savefig('.\\images\\house_price_vs_sqft_living.png')
