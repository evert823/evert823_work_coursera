import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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

def add_terms_of_polynomial(df, feature_name, degree):
    #Add columns polynomial terms as columns to dataframe till/including degree
    df_out = df.copy()

    for i in range(2, degree + 1):
        fn = f"{feature_name}_pwr_{i}"
        df_out[fn] = df_out[feature_name] ** i

    return df_out

def do_ridge(file_name, l2_penalty, degree, skip=False):
    if skip == True:
        return
    print(f"do_ridge {file_name} L2 {l2_penalty} degr. {degree}")
    house_data_train_df = read_house_data(path=path, file_name=file_name, dtype_dict=house_data_dtype_dict())

    x_feature_names = ['sqft_living']
    for i in range(2, degree + 1):
        x_feature_names.append(f'sqft_living_pwr_{i}')
    y_feature_names = ['price']

    #Create the dataframe with terms of polynomial of given degree
    polynomial_df = add_terms_of_polynomial(df=house_data_train_df, feature_name='sqft_living', degree=degree)

    #Sorting is important to make the plot to image working!!
    polynomial_df = polynomial_df.sort_values(['sqft_living', 'price'])

    #Create and fit a model
    y = polynomial_df[y_feature_names].values
    X = polynomial_df[x_feature_names].values

    #mylinreg = Ridge(alpha=l2_small_penalty)
    #mylinreg.fit(X, y)
    #w = np.hstack((mylinreg.intercept_.reshape(1, 1), mylinreg.coef_))
    mylinreg = make_pipeline(StandardScaler(), Ridge(alpha=l2_penalty))
    mylinreg.fit(X, y)
    a = mylinreg.named_steps['ridge']
    w = np.hstack((a.intercept_.reshape(1, 1), a.coef_))
    print(f"w with StandardScaler: {w} shape : {w.shape}")

    y_pred = mylinreg.predict(X)
    plt.figure(figsize=(16, 9), dpi=100)
    plt.plot(polynomial_df['sqft_living'],polynomial_df['price'],'.',
             polynomial_df['sqft_living'], y_pred,'-')
    plt.savefig(f".\\images\\{file_name}.png")

path = r"C:\Users\Evert Jan\courseradatascience\course02\module05\data\\"

#For question 1 - 4:
do_ridge(file_name="wk3_kc_house_train_data.csv",
         l2_penalty=1.5e-5,
         degree=15,
         skip=True)

#For question 5:
do_ridge(file_name="wk3_kc_house_set_1_data.csv",
         l2_penalty=1e-9,
         degree=15,
         skip=True)
do_ridge(file_name="wk3_kc_house_set_2_data.csv",
         l2_penalty=1e-9,
         degree=15,
         skip=True)
do_ridge(file_name="wk3_kc_house_set_3_data.csv",
         l2_penalty=1e-9,
         degree=15,
         skip=True)
do_ridge(file_name="wk3_kc_house_set_4_data.csv",
         l2_penalty=1e-9,
         degree=15,
         skip=True)

#For question 10:
do_ridge(file_name="wk3_kc_house_set_1_data.csv",
         l2_penalty=1.23e2,
         degree=15)
do_ridge(file_name="wk3_kc_house_set_2_data.csv",
         l2_penalty=1.23e2,
         degree=15)
do_ridge(file_name="wk3_kc_house_set_3_data.csv",
         l2_penalty=1.23e2,
         degree=15)
do_ridge(file_name="wk3_kc_house_set_4_data.csv",
         l2_penalty=1.23e2,
         degree=15)

