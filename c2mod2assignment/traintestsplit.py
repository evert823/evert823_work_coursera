import pandas as pd
import csv
from sklearn.model_selection import train_test_split

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

path = r"C:\Users\Evert Jan\courseradatascience\course02\module02\data\\"
outputpath = r"C:\Users\Evert Jan\courseradatascience\course02\module02\output\\"
file_name_all = "kc_house_data.csv"
file_name_train = "kc_house_train_data_v2.csv"
file_name_test = "kc_house_test_data_v2.csv"

# Read the full dataset
df = read_house_data(path=path, file_name=file_name_all, dtype_dict=house_data_dtype_dict())

# Split into train and test sets (e.g., 80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
# We hope that random_state has the same meaning as seed=0 in Turi Create

assess_dataframe(train_df)
assess_dataframe(test_df)

# Save the splits to CSV
#train_df.to_csv(outputpath + file_name_train, index=False, quoting=csv.QUOTE_NONNUMERIC)
#test_df.to_csv(outputpath + file_name_test, index=False, quoting=csv.QUOTE_NONNUMERIC)
train_df.to_csv(outputpath + file_name_train, index=False)
test_df.to_csv(outputpath + file_name_test, index=False)

# CONCLUSION Hard to reproduce the train testsplit the way that Coursera has done on sframe
