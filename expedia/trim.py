# orig_destination_distance contains nans
# date_time is string
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the data
train_df = pd.read_csv('input/train_1000.csv', header=0)
test_df = pd.read_csv('input/test_1000.csv', header=0)

def fillNansWithMean(dataframe):

    temp = []
    for x in dataframe:
        if (not np.isnan(x)):
            temp.append(x)
    mean = np.mean(temp)
    for i,x in np.ndenumerate(dataframe):
        if(np.isnan(x)):
            dataframe.set_value(i,mean,mean)



fillNansWithMean(train_df['orig_destination_distance'])
fillNansWithMean(test_df['orig_destination_distance'])
#transfromDatetimeToIntHours(train_df['date_time'])


f = lambda x: int(x[11:13])

train_df[['date_time']]= train_df[['date_time']].applymap(f)
print train_df