# coding: utf-8
__author__ = "Francesco Mosconi"
__copyright__ = "Copyright 2016, Data Weekends"
__license__ = "MIT"
__email__ = "info@dataweekends.com"

"""
Simple script detailing some of scikit-learn
functions including:
- train_test_split
- model fitting and evaluation
- decision tree classifier
- confusion matrix
"""


# Import the necessary libraries:
import pandas as pd

# Read data from Files
df = pd.read_csv('iris-2-classes.csv')
df['target'] = df['iris_type'].map({'virginica': 1, 'versicolor': 0})


# Define features (X) and target (y) variables
X = df[['sepal_length_cm', 'sepal_width_cm',
        'petal_length_cm', 'petal_width_cm']]
y = df['target']


# Initialize a decision tree model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth = 5, random_state=0)

#  Split the features and the target into a Train and a Test subsets.  
#  Ratio should be 70/30
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size = 0.3, random_state=0)


# Train the model
model.fit(X_train, y_train)

# Calculate the model accuracy score
my_score = model.score(X_test, y_test)

print "\n"
print "Using model: %s" % model
print "Classification Score: %0.2f" % my_score

# Print the confusion matrix for the decision tree model
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
print "\n=======confusion matrix=========="
print confusion_matrix(y_test, y_pred)


# ### 3) Iterate and improve
# 
# Now you have a basic pipeline. How can you improve the score? Try:
# - changing the parameters of the model
#   check the documentation here:
#   http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#   
# - changing the model itself
#   check examples here:
#   http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
#   
# - try separating 3 classes of flowers using the ```iris.csv``` dataset provided
