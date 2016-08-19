import numpy as np
from sklearn import cross_validation
import logistic_regressor as lr
from sklearn import linear_model
import scipy.io

######################################################################################
#   The sigmoid function                                                             #
#     Input: z: can be a scalar, vector or a matrix                                  #
#     Output: sigz: sigmoid of scalar, vector or a matrix                            #
######################################################################################

def sigmoid (z):
    sig = np.zeros(z.shape)
    # Your code here
    sig = 1/(1+np.exp(-z))
    # End your ode

    return sig

######################################################################################
#   The log_features transform                                                       #
#     Input: X: a data matrix                                                        #
#     Output: a matrix with every element x replaced by log(x+0.1)                   #
######################################################################################

def log_features(X):
    logf = np.zeros(X.shape)
    # Your code here
    logf = np.log(X+0.1)
    # End your ode
    return logf

######################################################################################
#   The std_features transform                                                       #
#     Input: X: a data matrix                                                        #
#     Output: a matrix with every column with zero mean, unit variance               #
######################################################################################

def std_features(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

######################################################################################
#   The bin_features transform                                                       #
#     Input: X: a data matrix                                                        #
#     Output: a matrix with every element x replaced by 1 if x > 0 else 0            #
######################################################################################

def bin_features(X):
    tX = np.zeros(X.shape)
    # your code here
    tX[X>0] = 1
    # end your code
    return tX

######################################################################################
#   The select_lambda_crossval function                                              #
#     Inputs: X: a data matrix                                                       #
#             y: a vector of labels                                                  #
#             lambda_low, lambda_high,lambda_step: range of lambdas to sweep         #
#             penalty: 'l1' or 'l2'                                                  #
#     Output: best lambda selected by crossvalidation for input parameters           #
######################################################################################

# Select the best lambda for training set (X,y) by sweeping a range of
# lambda values from lambda_low to lambda_high in steps of lambda_step
# pick sklearn's LogisticRegression with the appropriate penalty (and solver)

# For each lambda value, divide the data into 10 equal folds
# using sklearn's cross_validation KFold function.
# Then, repeat i = 1:10:
#  1. Retain fold i for testing, and train logistic model on the other 9 folds
#  with that lambda
#  2. Evaluate accuracy of model on left out fold i
# Accuracy associated with that lambda is the averaged accuracy over all 10
# folds.
# Do this for each lambda in the range and pick the best one
#
def select_lambda_crossval(X,y,lambda_low,lambda_high,lambda_step,penalty):
   
    
    best_lambda = lambda_low

    # Your code here
    # Implement the algorithm above.

    lambda_test = lambda_low
    highest_accuracy =0.0
    
    while(lambda_test<=lambda_high):       
        sk_logreg_l1 = linear_model.LogisticRegression(C=1.0/lambda_test,solver='liblinear',fit_intercept=False,penalty='l1')
        sk_logreg_l2 = linear_model.LogisticRegression(C=1.0/lambda_test,solver='sag',fit_intercept=False)       
        if (penalty == "l1"):
            sk_logreg = sk_logreg_l1
        elif (penalty == "l2"):
            sk_logreg = sk_logreg_l2
        else:
            print "error in penalty"  
            break                     
        kf = cross_validation.KFold(X.shape[0], n_folds = 10)        
        cumulate_accuracy = 0.0
        for train_set,test_set in kf:
            X_train, X_test = X[train_set], X[test_set]
            y_train,y_test = y[train_set],y[test_set]
            sk_logreg.fit(X_train,y_train)
            #print "Theta found by sklearn with ",penalty," regularization " ,sk_logreg.coef_[0]
            y_predict = sk_logreg.predict(X_test)
            sub_accuracy = 1- np.nonzero(np.round(y_predict - y_test))[0].size/float(y_test.shape[0])
            #print "The accuracy is ", sub_accuracy
            cumulate_accuracy = cumulate_accuracy+sub_accuracy
            #print "cumulate accuracy is ", cumulate_accuracy
        accuracy = cumulate_accuracy/10
        if(accuracy>highest_accuracy):
            highest_accuracy = accuracy
            best_lambda = lambda_test        
        lambda_test = lambda_test + lambda_step
        

    # end your code

    return best_lambda


######################################################################################

def load_mat(fname):
  d = scipy.io.loadmat(fname)
  Xtrain = d['Xtrain']
  ytrain = d['ytrain']
  Xtest = d['Xtest']
  ytest = d['ytest']

  return Xtrain, ytrain, Xtest, ytest


def load_spam_data():
    data = scipy.io.loadmat('spamData.mat')
    Xtrain = data['Xtrain']
    ytrain1 = data['ytrain']
    Xtest = data['Xtest']
    ytest1 = data['ytest']

    # need to flatten the ytrain and ytest
    ytrain = np.array([x[0] for x in ytrain1])
    ytest = np.array([x[0] for x in ytest1])
    return Xtrain,Xtest,ytrain,ytest

    
