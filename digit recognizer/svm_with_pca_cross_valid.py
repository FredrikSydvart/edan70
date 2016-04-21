import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from numpy import genfromtxt, savetxt

# Cross valid score
from sklearn.cross_validation import cross_val_score

# Matplot to plot
import matplotlib.pyplot as plt
#%matplotlib inline

CROSS_FOLDS = 5

# PCA components
COMPONET_NUM = 35

def main():
    # The competition datafiles are in the directory ../input
    # Read competition data files:
    train = pd.read_csv("input/train.csv")
    test  = pd.read_csv("input/test.csv")

    # Write to the log:
    print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
    print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
    # Any files you write to the current directory get shown as outputs

    x_train = train.values[:,1:].astype(float)
    y_train = train.values[:,0]

    pca = PCA(n_components=COMPONET_NUM,whiten=True)
    pca.fit(x_train)
    x_train = pca.transform(x_train)

    # Search for an optimal value of estimates for SVC
    gamma_range = np.arange(0.01, 0.13, 0.01)
    k_scores = []
    for k in gamma_range:
        print "Trying gamma nbr " + str(k)
        svc = SVC(gamma=k)
        scores = cross_val_score(svc, x_train, y_train, cv=CROSS_FOLDS, scoring='accuracy')
        k_scores.append(scores.mean())
    print k_scores

    # Round for nicer matplot
    for index, k in enumerate(k_scores):
        k_scores[index] = round(k_scores[index], 4)

    # Plot the result and save figure
    plt.plot(gamma_range, k_scores)
    plt.xlabel('Value of estimators for SVC classifier')
    plt.ylabel('Cross-Validated Accuracy')
    plt.savefig('svm_with_pca_cross_valid_5.png')
    plt.show()


    # subm=np.empty((len(predict),2))
    # subm[:,0]=np.arange(1,len(predict)+1)
    # subm[:,1]=predict
    # np.savetxt('submission.csv',subm,fmt='%d',delimiter=',',header='ImageId,Label',comments='')

if __name__=="__main__":
    main()