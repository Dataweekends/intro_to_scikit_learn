# coding: utf-8
__author__ = "Francesco Mosconi"
__copyright__ = "Copyright 2016, Data Weekends"
__license__ = "MIT"
__email__ = "info@dataweekends.com"

"""
General purpose script detailing some of scikit-learn most common
functions including:
- data loading
- several types of classifiers
- preprocessing
- pipelines
- cross validation
- grid search
"""

# Load a dataset from one of those available in sklearn:
print "Simple starter scikit-learn example"
print


from sklearn.datasets import load_iris
data = load_iris()
# for other datasets see:
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets


# Define features (X) and target (y) variables
X = data['data']
y = data['target']


# Initialize a decision tree model
from sklearn.tree import DecisionTreeClassifier
estimator = DecisionTreeClassifier(max_depth=5, random_state=0)

# other ideas for classifiers:
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier


# preprocess the data using a standard scaler
from sklearn.preprocessing import StandardScaler
transformer = StandardScaler()


# combine the steps in a pipeline:
# step1: transformer
# step2: model
# a pipeline can contain N transformer steps and an optional
# final estimator step
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(transformer,
                         estimator)


# Do a cross validation:
from sklearn.model_selection import cross_val_score
cvscores = cross_val_score(pipeline, X, y, n_jobs=-1)

print "The pipeline CV score is:"
print cvscores.mean().round(3), "+/-", cvscores.std().round(3)
print

# a pipeline exposes the API of a transformer and
# estimator combined, so all the usual fit, predict and
# transform methods are available.
# if you're using text data or other data you could include
# here preprocessing steps like:
# - feature extraction
# - feature selection
# - normalization
# - etc.

# if your model has hyperparameters, try using grid search
# to optimize them:
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
model = SVC()
params = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
gsmodel = GridSearchCV(model, params, n_jobs=-1)
gsmodel.fit(X, y)
print "Best Model:"
print gsmodel.best_estimator_
print
print "Best Parameters:"
print gsmodel.best_params_
print
print "Best Score:"
print gsmodel.best_score_


# Exercises and ideas:
# - benchmark your results against a dummy classifier
# - choose a different datasets
# - choose features with feature selection
# - build a custom trasformer class
# - build a custom classifier class
# - choose an example from http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
#   and try to reproduce it
# - use feature extraction to classify unstructured data
# - apply to a dataset of your choice

