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

path = r"C:\Users\Evert Jan\courseradatascience\course02\module03\data\\"
file_name_inp = "kc_house_train_data.csv"

x_feature_names = ['sqft_living', 'sqft_living15']
y_feature_names = ['price']

house_data_df = read_house_data(path=path, file_name=file_name_inp, dtype_dict=house_data_dtype_dict())

#Cross-check with sklearn
y2 = house_data_df[y_feature_names].values
X2 = house_data_df[x_feature_names].values
mylinreg = LinearRegression()
mylinreg.fit(X2, y2)
coeffs = dict(zip(x_feature_names, mylinreg.coef_))
coeffs['intercept'] = mylinreg.intercept_
print(f"par + intercept model1 : {coeffs}")
