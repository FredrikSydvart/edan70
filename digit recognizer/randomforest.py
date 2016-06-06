from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt
import numpy as np

def main():
    # create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('input/train.csv','r'), delimiter=',', dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('input/test.csv','r'), delimiter=',', dtype='f8')[1:]
    
    # create and train the random forest
    rf = RandomForestClassifier(n_estimators=100, n_jobs=2)

    #n_samples = len(train)
    #train_data = train[0].reshape((n_samples, -1))
    #test_data = test.reshape((n_samples, -1))

    rf.fit(train, target)

    # result from prediction
    result = rf.predict(test).astype(int)

    with open('randomforest_submission.csv', 'w') as f:
        f.write("ImageId,Label\n")
        for index, value in enumerate(np.nditer(result)):
            f.write(str(index + 1) + "," + str(value) + "\n")

if __name__=="__main__":
    main()