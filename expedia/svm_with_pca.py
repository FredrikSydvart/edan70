import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from numpy import genfromtxt, savetxt

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

np.set_printoptions(threshold=np.nan)


# PCA components
COMPONET_NUM = 35

def get5Best(x):
    result = []
    for z in x.argsort()[::-1][:5]:
        if z!=0:
            result.append(z)
    return " ".join([str(int(z)) for z in result])

def getBest(predicts, cluster_ids):
	result = []
	for predict in predicts:
		# Sort and reverse and take the 5 best
		predict = predict.sort()
		predict = predict[::-1]
		predict = predict[:5]

		# Map to the cluser id





def main():
	# The competition datafiles are in the directory ../input
	# Read competition data files:
	train = pd.read_csv("input/train_100.csv")
	test  = pd.read_csv("input/test_100.csv")

	# Write to the log:
	print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
	print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
	# Any files you write to the current directory get shown as outputs

	#x_train = train.values[:,1:].astype(float)
	#x_train = train.values[:,1:]
	#y_train = train.values[:,0]


	#x_train = x_train.values[1:10] + x_train.values[12:]
	#x_train = train[['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'orig_destination_distance', 'user_id', 'is_mobile', 'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market', 'is_booking', 'cnt']]
	x_train = train[['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'orig_destination_distance', 'user_id', 'is_mobile', 'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market']]
	x_train = x_train.fillna(0)
	#print x_train
	
	y_train = train[['hotel_cluster']]
	y_train = y_train.fillna(0)
	#print y_train
	#x_train = train.values[1:10] + train.values[12:0]

	test = test[['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'orig_destination_distance', 'user_id', 'is_mobile', 'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market']]
	test = test.fillna(0)

	#pca = PCA(n_components=COMPONET_NUM,whiten=True)
	#pca.fit(x_train)
	#x_train = pca.transform(x_train)

	# Run SVC on training data
	print "Train svm"
	svc = SVC(gamma=0.05, probability=True)
	#rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
	#svc = BaggingClassifier(rf, n_estimators=2, n_jobs=2)
	svc.fit(x_train, y_train.values.ravel())

	# Predict with testing data
	#test = pca.transform(test)
	predict = svc.predict_proba(test)

	# Save to file
	print predict
	#subm = np.empty((len(predict),2))
	#subm[:,0] = np.arange(1,len(predict)+1)
	#subm[:,1] = predict
	#np.savetxt('submission.csv',subm,fmt='%d',delimiter=',',header='id,hotel_cluster',comments='')

	print get5Best(predict)
	#print np.apply_along_axis(get5Best, 1, predict)
	#submit = pd.read_csv('submission.csv')
	#submit['hotel_cluster'] = np.apply_along_axis(get5Best, 1, predict)
	#submit.head()
	#submit.to_csv('submission_20160418_ent_1.csv', index=False)

if __name__=="__main__":
    main()