# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
import numpy
#import Sequential Neural net from Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"

n_students = len(student_data.index)

# TODO: Calculate number of features
n_features = len(student_data.columns) - 1

# TODO: Calculate passing students
n_passed = len(student_data[student_data['passed'] == 'yes'].index)
# TODO: Calculate failing students
n_failed = len(student_data[student_data['passed'] == 'no'].index)

# TODO: Calculate graduation rate
grad_rate = np.divide(float(n_passed), n_students) * 100

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1]

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()


def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix=col)

            # Collect the revised columns
        output = output.join(col_data)

    return output


X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features.as_matrix())
    #y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    #return f1_score(target.values, y_pred, pos_label = 'yes')
    return f1_score(target, y_pred, pos_label = 1)


def convoluted_neuralnet(optimizer='rmsprop', init='glorot_uniform'):
    ''' Implimenting dense convulated network with Keras library'''

    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=48, init=init, activation='relu'))
    model.add(Dense(8, init=init, activation='relu'))
    model.add(Dense(1, init=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def custom_scorer(label, prediction):
    return f1_score(label, prediction, pos_label=1)


def fine_tune_model():
# grid search epochs, batch size and optimizer
    optimizers = ['rmsprop', 'adam']
    init = ['glorot_uniform', 'normal', 'uniform']
    epochs = numpy.array([50, 100, 150])
    batches = numpy.array([5, 10, 20])

# TODO: Create the parameters list you wish to tune
    parameters = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)

# TODO: Initialize the classifier
    clf = KerasClassifier(build_fn=convoluted_neuralnet)

# TODO: Make an f1 scoring function using 'make_scorer'
    f1_scorer = make_scorer(custom_scorer,  greater_is_better=True)

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
    grid_obj = GridSearchCV(estimator=clf, param_grid=parameters, verbose=0, scoring=f1_scorer)

# Hot encode Yes,No to 1,0 else the library would throw error..
    y_encoded = pd.DataFrame(y_all).replace(['yes', 'no'], [1, 0])

    X_train,X_test,y_train,y_test = train_test_split(X_all, y_encoded, train_size=300, test_size=95,
                                                                stratify=y_encoded, random_state= 42)

# TODO: Fit the grid search object to the training data and find the optimal parameters
    grid_obj = grid_obj.fit(X_train.as_matrix(), y_train.as_matrix())

# Get the estimator
    clf = grid_obj.best_estimator_

# Output the best parameters for the data-set as per GridSerachCV
    print "The tuned parameters from GridSearchCV for the dataset: ", grid_obj.best_params_

# Print the F1 score for train and test set...
    print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))

if __name__ == '__main__':
    fine_tune_model()