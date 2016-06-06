import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

np.set_printoptions(threshold=np.nan)

def get5Best(x):
    result = []
    for z in x.argsort()[::-1]:
        if np.any(z):
            result.append(z)
    return np.asarray(result)

def main():
    # The competition datafiles are in the directory /input
    # Read competition data files:
    train_csv = pd.read_csv("input/train.csv")
    test_csv  = pd.read_csv("input/test.csv")

    # Prepare train by taking columns and filling NaNs
    #train = train[['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'orig_destination_distance', 'user_id', 'is_mobile', 'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market', 'is_booking', 'cnt']]
    train = train_csv[['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'orig_destination_distance', 'user_id', 'is_mobile', 'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market']]
    train = train.fillna(0)
    target = train_csv[['hotel_cluster']]
    target = target.fillna(0)

    # Prepare test by taking columns and filling NaNs
    test = test_csv[['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'orig_destination_distance', 'user_id', 'is_mobile', 'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market']]
    test = test.fillna(0)

    # Run Random Forest on training data
    print "Training Random Forest"
    rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    svc = BaggingClassifier(rf, n_estimators=2, n_jobs=2)
    svc.fit(train, target.values.ravel())

    # Predict with testing data
    print "Predicting test data"
    predict = svc.predict_proba(test)

    # Generate submission
    print "Generating submission"
    stuff = np.apply_along_axis(get5Best, 1, predict)
    subm = np.empty((len(predict),6))
    subm[:,0] = np.arange(1,len(predict)+1)
    subm[:,1] = stuff[:,0]
    subm[:,2] = stuff[:,1]
    subm[:,3] = stuff[:,2]
    subm[:,4] = stuff[:,3]
    subm[:,5] = stuff[:,4]
    np.savetxt('random_forest_submission.csv',subm,fmt='%d,%d %d %d %d %d',delimiter=',',header='id,hotel_cluster',comments='')

    print "Done"

if __name__=="__main__":
    main()