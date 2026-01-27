import pandas as pd
from math import log, sqrt

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

def add_columns(df):
    df['floors'] = df['floors'].astype(float)  # Convert floors to float
    df['sqft_living_sqrt'] = df['sqft_living'].apply(sqrt)
    df['sqft_lot_sqrt'] = df['sqft_lot'].apply(sqrt)
    df['bedrooms_square'] = df['bedrooms']*df['bedrooms']
    df['floors_square'] = df['floors']*df['floors']
    return df

path = r"C:\Users\Evert Jan\courseradatascience\course02\module06\data\\"

house_data_mock_df = read_house_data(path=path, file_name="mockdata.csv", dtype_dict=house_data_dtype_dict())
house_data_mock_df = add_columns(df=house_data_mock_df)
assess_dataframe(df=house_data_mock_df)

house_data_all_df = read_house_data(path=path, file_name="kc_house_data.csv", dtype_dict=house_data_dtype_dict())
house_data_all_df = add_columns(df=house_data_all_df)
assess_dataframe(df=house_data_all_df)
