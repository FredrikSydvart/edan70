from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt

# Cross valid score
from sklearn.cross_validation import cross_val_score

# Matplot to plot
import matplotlib.pyplot as plt
#%matplotlib inline

CROSS_FOLDS = 5


def main():
    # create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('input/train.csv','r'), delimiter=',', dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]

    # Search for an optimal value of estimates for RandomForestClassifier
    estimate_range = range(80, 220,5)
    k_scores = []
    for k in estimate_range:
        print "Trying estimator nbr " + str(k)
        rf = RandomForestClassifier(n_estimators=k, n_jobs=2)
        scores = cross_val_score(rf, train, target, cv=CROSS_FOLDS, scoring='accuracy')
        k_scores.append(scores.mean())
    print k_scores

    # Round for nicer matplot
    for index, k in enumerate(k_scores):
        k_scores[index] = round(k_scores[index], 4)

    # Plot the result and save figure
    plt.plot(estimate_range, k_scores)
    plt.xlabel('Value of estimators for RandomForestClassifier')
    plt.ylabel('Cross-Validated Accuracy')
    plt.savefig('randomforest_cross_valid_5.png')
    plt.show()

if __name__=="__main__":
    main()