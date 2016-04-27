import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from numpy import genfromtxt, savetxt
import ml_metrics as metrics

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.cross_validation import cross_val_score

CROSS_FOLDS = 5

np.set_printoptions(threshold=np.nan)

def map5scorer(estimator, train, target):
    print ("lewl")
    prediction = estimator.predict_proba(train)
    print prediction
    print target
    map5k = metrics.mapk(target, prediction)
    print(map5k)
    return map5k

def get5Best(x):
    result = []
    for z in x.argsort()[::-1][:5]:
        if z != 0:
            result.append(z)
    #stuff = np.array([])
    #return np.concatenate(([stuff, [str(int(z)) for z in result]]))
    #return np.asarray(result)
    return " ".join([str(int(z)) for z in result])

def main():
    # The competition datafiles are in the directory /input
    # Read competition data files:
    #train = pd.read_csv("input/train_1000.csv")
    #train_chunk = pd.read_csv('input/train.csv', chunksize=100000)

    # Training cols
    print ("Loading training csv.")
    #train_cols = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'orig_destination_distance', 'user_id', 'is_mobile', 'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market', 'hotel_cluster']
    train_cols = ['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'hotel_market', 'hotel_country', 'hotel_cluster']
    train = pd.DataFrame(columns=train_cols)
    train_chunk = pd.read_csv('train_100.csv', chunksize=10)
    print ("Training csv loaded.")

    # Read each chunk to train
    for chunk in train_chunk:
        #train = pd.concat( [ train, chunk ] )
        train = pd.concat( [ train, chunk[chunk['is_booking']==1][train_cols] ] )

    training = genfromtxt(open('train_100.csv', 'r'), delimiter=',', dtype='f8')[1:]
    x_train = [x[:-1] for x in training]
    x_train = np.nan_to_num(x_train)
    y_train = [x[-1:] for x in training]
    y_train = np.nan_to_num(y_train)
    y_train = y_train.ravel()
    print training
    print x_train
    print y_train

    #stuff = np.array([])
    # stuff = []
    # for nbr in y_train:
    #     temp = [nbr]
    #     #temp = np.array([nbr])
    #     #stuff = np.concatenate([stuff, temp])
    #     stuff.append(temp)
    # y_train = stuff
    # print y_train



    # Load each column
    #train.head()
    #x_train = train[['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'orig_destination_distance', 'user_id', 'is_mobile', 'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market']].values
    #x_train = train[['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'hotel_market', 'hotel_country']].values
    #x_train = np.nan_to_num(x_train)
    #x_train = x_train.fillna(0)
    #y_train = train['hotel_cluster'].values
    #y_train = y_train.fillna(0)
    #y_train = np.nan_to_num(y_train)

    # Run RandomForest on training data
    print ("Training RandomForest.")
    #svc = SVC(gamma=0.05, probability=True)
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=4)
    bclf = BaggingClassifier(rf, n_estimators=2, n_jobs=4)
    bclf.fit(x_train, y_train)
    print ("Training done.")

    scores = cross_val_score(rf, x_train, y_train, cv=CROSS_FOLDS, scoring=map5scorer)
    print scores
    #print get5Best(predict)
    #print np.apply_along_axis(get5Best, 1, predict)

if __name__=="__main__":
    main()

