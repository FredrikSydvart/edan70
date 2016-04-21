from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import numpy as np

def main():
    # create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('input/train.csv','r'), delimiter=',', dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('input/test.csv','r'), delimiter=',', dtype='f8')[1:]
    
    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=2)

    n_samples = len(train)
    #train_data = train[0].reshape((n_samples, -1))
    #test_data = test.reshape((n_samples, -1))

    rf.fit(train, target)

    # result from prediction
    result = rf.predict(test).astype(int)

    # array from 0 to 28000
    i = np.arange(1, 28001);

    with open('randomforest_submission.csv', 'w') as file:
    	file.write("ImageId,Label\n")
    	for index, value in enumerate(np.nditer(result)):
    		file.write(str(index + 1) + "," + str(value) + "\n")


    #savetxt('submission2.csv', np.hstack([i, result]), delimiter=',', fmt='%u', header="\"ImageId\", \"Label\"")

if __name__=="__main__":
    main()