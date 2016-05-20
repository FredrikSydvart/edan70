import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from numpy import genfromtxt, savetxt
import ml_metrics as metrics

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import GradientBoostingClassifier

CROSS_FOLDS = 4
validate = 0 # 1 for cross volds, 0 for submission

np.set_printoptions(threshold=np.nan)

def map5scorer(estimator, train, target):
    #print ("lewl")
    prediction = estimator.predict_proba(train)
    #print prediction
    #print target
    map5k = metrics.mapk(target, prediction)
    #print(map5k)
    return map5k

#def map5eval(estimator, preds, dtrain):
def map5eval(estimator, train, target):
    prediction = estimator.predict_proba(train)
    actual = target
    predicted = prediction.argsort(axis=1)[:,-np.arange(1,6)]
    metric = 0.
    for i in range(5):
        metric += np.sum(actual==predicted[:,i])/(i+1)
    metric /= actual.shape[0]
    return metric

    #actual = dtrain.get_label()
    #predicted = preds.argsort(axis=1)[:,-np.arange(1,6)]
    #metric = 0.
    #for i in range(5):
    #    metric += np.sum(actual==predicted[:,i])/(i+1)
    #metric /= actual.shape[0]
    #return metric
    #return 'MAP@5', metric

def fillNansWithMean(dataframe):

    temp = []
    for x in dataframe:
        if (not np.isnan(x)):
            temp.append(x)
    mean = np.mean(temp)
    for i,x in np.ndenumerate(dataframe):
        if(np.isnan(x)):
            dataframe.set_value(i,mean,mean)


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
    #train_cols = ['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'hotel_market', 'hotel_country', 'hotel_cluster']
    #train = pd.DataFrame(columns=train_cols)
    #train_chunk = pd.read_csv('input/train_1000.csv', chunksize=10)
    #print ("Training csv loaded.")

    # Read each chunk to train
    #for chunk in train_chunk:
        #train = pd.concat( [ train, chunk ] )
    #    train = pd.concat( [ train, chunk[chunk['is_booking']==1][train_cols] ] )

    training = genfromtxt(open('input/train_1000.csv', 'r'), delimiter=',', dtype='f8')[1:]
    # Remove is_booking
    x_train = [x[:-1] for x in training]
    x_train = np.nan_to_num(x_train)
    y_train = [x[-1:] for x in training]
    y_train = np.nan_to_num(y_train)
    y_train = y_train.ravel()

    # Newest code
    #train = pd.read_csv('input/train_10.csv', header=0)
    #fillNansWithMean(train['orig_destination_distance'])

    # date time
    f = lambda x: int(x[11:13])

    # check in and out
    d = lambda x: int(x[8:10])

    #train[['date_time']] = train[['date_time']].applymap(f)
    #train[['srch_ci']] = train[['srch_ci']].applymap(d)
    #train[['srch_co']] = train[['srch_co']].applymap(d)

    #x_train = [x[:-1] for x in train]
    #y_train = [x[-1:] for x in train]

    #y_train = train['hotel_cluster'].values
    #x_train = train.drop('hotel_cluster', 1)
    #print x_train
    #print y_train


    #print training
    #print x_train
    #print y_train

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

    #svc = SVC(gamma=0.05, probability=True)

    # Random Forest
    #for n in range(1,4):
        # Random Forest
        #print ("Training RandomForest.")
        #trees = 50 + (n * 10)
        #rf = RandomForestClassifier(n_estimators=trees, max_depth=10, n_jobs=4)
        #bclf = BaggingClassifier(rf, n_estimators=2, n_jobs=4)
        #bclf.fit(x_train, y_train)

        #print ("Training done.")

        #scores = cross_val_score(xgb2, x_train, y_train, cv=CROSS_FOLDS, scoring=map5eval)
        #print "Estimator: " + str(estimator)
        #print str(n) + " " + str(scores)
        # The mean score and the 95% confidence interval of the score estimate are hence given by:
        #print("Map5k: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        #print "\n"

    # Gradient Boosting
    if validate == 1:
        print ("Training GradientBoosting and validating.")
        for n in range(1, 4):
            for m in range(1, 4):
                for o in range(1, 2):
                    # Boosting settings
                    depth = 3 + o
                    estimator = 10 + (n * 5)
                    rate = 0.05 + (m * 0.01)

                    #print ("Training XGBoost")
                    gb = GradientBoostingClassifier(max_depth=depth, n_estimators=estimator, learning_rate=rate)
                    gb.fit(x_train, y_train)
                    #print ("Training done.")
                    scores = cross_val_score(gb, x_train, y_train, cv=CROSS_FOLDS, scoring=map5eval)
                    print "------------------------"
                    print "Depth: \t" + str(depth)
                    print "Estimator: \t" + str(estimator)
                    print "Rate: \t" + str(rate)
                    print "Score: \t" + str(scores)
                    print "Mean: \t" + str(scores.mean())
                    # The mean score and the 95% confidence interval of the score estimate are hence given by:
                    print("Map5k: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
                    print "------------------------"
                    print "\n"

    # Make predictions and submission
    if validate == 0:
        print ("Training GradientBoosting.")
        # Fit the classifier to the training data
        # Gradient Boosting
        depth = 4
        estimator = 20
        rate = 0.06
        gb = GradientBoostingClassifier(max_depth=depth, n_estimators=estimator, learning_rate=rate)
        gb.fit(x_train, y_train)
        print ("Training done.")

        print ("Begin predicting.")
        predict = np.array([])
        test_chunk = pd.read_csv('input/test_1000.csv', chunksize=2, header=0)
        # Read each chunk to test
        i = 0
        for chunk in enumerate(test_chunk):
            # Take values from chunk
            test = chunk[1]

            # date time
            f = lambda x: int(x[11:13])

            # check in and out
            d = lambda x: int(x[8:10])

            # Empty
            e = lambda x: int(0)

            # Convert dates to ints
            test[['date_time']] = test[['date_time']].applymap(f)
            test[['srch_ci']] = test[['srch_ci']].applymap(d)
            test[['srch_co']] = test[['srch_co']].applymap(d)

            # Fill is_booking with 0 to have nbr of features the same in the model
            test[['is_booking']] = test[['srch_co']].applymap(e)


            # BUG: Might contain NaN somewhere
            fillNansWithMean(test['site_name'])
            # fillNansWithMean(test['date_time'])
            fillNansWithMean(test['site_name'])
            fillNansWithMean(test['posa_continent'])
            fillNansWithMean(test['user_location_country'])
            fillNansWithMean(test['user_location_region'])
            fillNansWithMean(test['user_location_city'])
            fillNansWithMean(test['orig_destination_distance'])
            fillNansWithMean(test['user_id'])
            fillNansWithMean(test['is_mobile'])
            fillNansWithMean(test['is_package'])
            fillNansWithMean(test['channel'])
            # fillNansWithMean(test['srch_ci'])
            # fillNansWithMean(test['srch_co'])
            fillNansWithMean(test['srch_adults_cnt'])
            fillNansWithMean(test['srch_children_cnt'])
            fillNansWithMean(test['srch_rm_cnt'])
            fillNansWithMean(test['srch_destination_id'])
            fillNansWithMean(test['srch_destination_type_id'])
            fillNansWithMean(test['hotel_continent'])
            fillNansWithMean(test['hotel_country'])
            fillNansWithMean(test['hotel_market'])


            # Make prediction
            proba = gb.predict_proba(test)

            if i > 0:
                predict = np.concatenate([predict, proba])
            else:
                predict = proba
            print ("Chunk id: " + str(i))
            i += 1
        print ("Predicting done.")

        print ("Generating submission")
        submit = pd.read_csv('sample_submission.csv')
        submit['hotel_cluster'] = np.apply_along_axis(get5Best, 1, predict)
        submit.head()
        submit.to_csv('submission_gradientboosting_crossfolds.csv', index=False)

    print ("Done.")
    #print get5Best(predict)
    #print np.apply_along_axis(get5Best, 1, predict)

if __name__=="__main__":
    main()

