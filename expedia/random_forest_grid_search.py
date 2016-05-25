import pandas as pd
import numpy as np
from numpy import genfromtxt
import ml_metrics as metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

CROSS_FOLDS = 4

np.set_printoptions(threshold=np.nan)

# Uses the Mean Average Precision at 5 (MAP@5) evaluation
def map5eval(estimator, train, target):
    prediction = estimator.predict_proba(train)
    actual = target
    predicted = prediction.argsort(axis=1)[:,-np.arange(1,6)]
    metric = 0.
    for i in range(5):
        metric += np.sum(actual==predicted[:,i])/(i+1)
    metric /= actual.shape[0]
    return metric

def main():
    # The competition datafiles are in the directory /input

    # Training cols
    print ("Loading training csv.")
    training = genfromtxt(open('input/train_10000.csv', 'r'), delimiter=',', dtype='f8')[1:]
    # All columns except target
    train = [x[:-1] for x in training]
    train = np.nan_to_num(train)

    # Only target column
    target = [x[-1:] for x in training]
    target = np.nan_to_num(target)
    target = target.ravel()
    print ("Loading done.")

    print ("Training Random Forest and doing Grid search with cross folds.")
    # Generate parameters to search for
    tuned_parameters = []
    for n in range(250,700,50):
        params = {'n_estimators': [n], 'max_depth': [None], 'n_jobs': [2]}
        tuned_parameters.append(params)

    # Set the parameters for random forest to be tested
    #tuned_parameters = [{'n_estimators': [2], 'max_depth': [4], 'n_jobs': [2]},
    #                    {'n_estimators': [3], 'max_depth': [4], 'n_jobs': [2]}]

    # Create Random Forest estimator model
    randomforest  = RandomForestClassifier()

    # Create GridSearch with params and fit to training
    gridsearch    = GridSearchCV(estimator=randomforest, param_grid=tuned_parameters, cv=CROSS_FOLDS, scoring=map5eval)
    gridsearch.fit(train, target)
    print ("Searching done.")

    # Summarize the results of the grid search
    print ("\n------------------------")
    print ("Best score: \t" + str(gridsearch.best_score_))
    print ("Best params: \t" + str(gridsearch.best_params_))
    print ("------------------------")
    # The mean score, the 95% confidence interval and the scores are printed
    for params, mean_score, scores in gridsearch.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    print ("------------------------")

    print ("Done.")

if __name__=="__main__":
    main()

