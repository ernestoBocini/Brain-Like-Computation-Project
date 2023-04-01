import numpy as np
from exploration import transform_classes_to_int, transform_to_8_classes
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold


def compute_mse(y, tx, w):
    """
    Compute the Mean Square Error as defined in class.
    Takes as input the targeted y, the sample matrix X and the feature fector w.
    """
    e = y - tx@w
    mse = e.T.dot(e) /(2*len(e))
    return mse


def ridge_regression(y, tx, lambda_):
    """
    Compute an esimated solution of the problem y = tx @ w , and the associated error. Note that this method
    is a variant of least_square() but with an added regularization term lambda_. 
    This method is equivalent to the minimization problem of finding w such that |y-tx@w||^2 + lambda_*||w||^2 is minimal. 
    The error is the mean square error of the targeted y and the solution produced by the least square function.
    Takes as input the targeted y, the sample matrix X and the regulariation term lambda_.
    """
    x_t = tx.T
    lambd = lambda_ * 2 * len(y)
    w = np.linalg.solve (np.dot(x_t, tx) + lambd * np.eye(tx.shape[1]), np.dot(x_t,y)) 
    loss = compute_mse(y, tx, w)

    return w,loss




def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_objects(spikes, stimulus, objects, candidates, k):
    """
    Completes k-fold cross-validation for Least Squares with GD, SGD, Normal Equations, Logistic and Regularized Logistic 
    Regression with SGD
    """

    # transform objects to 8 classes and to int
    objects = transform_classes_to_int(transform_to_8_classes(objects))

    # concatenate the 2 arrays
    # convert objects to numpy array
    objects = np.array(objects).reshape(-1, 1)
    print("objects shape: ", objects.shape)
    print("stimulus shape: ", stimulus.shape)
    x = np.concatenate((stimulus, objects), axis=1)

    # Split the training set in subsets according to the obj value 

    obj_class = {
        0: x[:, -1] == 0,
        1: x[:, -1] == 1,
        2: x[:, -1] == 2, 
        3: x[:, -1] == 3,
        4: x[:, -1] == 4,
        5: x[:, -1] == 5,
        6: x[:, -1] == 6,
        7: x[:, -1] == 7
        }
    
    for idx in range(len(obj_class)):
        x_class = x[obj_class[idx]]
        y_class= spikes[obj_class[idx]]

        kf = KFold(n_splits=k)
        
        model = Ridge(fit_intercept=True)

         # Create a parameter grid
        param_grid = {'alpha': candidates}

        # Initialize a GridSearchCV object
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='explained_variance')

        # Perform the grid search
        grid_search.fit(x_class, y_class)

        mse = grid_search.best_score_

        # Print the best alpha value
        print("Best alpha value for class {0} is {1} with mse {2}: ". format(idx, grid_search.best_params_['alpha'], mse))



def select_parameters_ridge_regression_obj(y,tX,lambdas,k_fold,seed):
    """
    Given the training set and a set of tuples of parameters (alphas, lamdas, degrees) 
    for each jet_subset returns the tuple which maximize the accuracy predicted through Cross Validation 
    """

    par_lamb = []
    accus = []

    # Split the training set in subsets according to the jet value 
    msk_obj = {
        0: tX[:, -1] == 0,
        1: tX[:, -1] == 1,
        2: tX[:, -1] == 2, 
        3: tX[:, -1] == 3,
        4: tX[:, -1] == 4,
        5: tX[:, -1] == 5,
        6: tX[:, -1] == 6,
        7: tX[:, -1] == 7
        }

    for idx in range(len(msk_obj)):
        tx = tX[msk_obj[idx]]
        ty = y[msk_obj[idx]]
        
        lambd,accu = select_parameters_ridge_regression(lambdas, k_fold, ty, tx, seed)
        par_lamb.append(lambd)
        accus.append(accu)

    return par_lamb, accus


def select_parameters_ridge_regression(lambdas, k_fold, y, tx, seed):
    """
    Given the training set and a set of tuples of parameters (alphas, lamdas, degrees) 
    returns the tuple which maximize the accuracy predicted through Cross Validation 
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    comparison = []

    for lamb in lambdas:
                accs_test = []
                for k in range(k_fold):
                        _, acc_test = cross_validation(y, tx, ridge_regression, k_indices, k, lamb)
                        accs_test.append(acc_test)
                comparison.append([lamb,np.mean(accs_test)])
    
    comparison = np.array(comparison)
    ind_best =  np.argmax(comparison[:,1]) 
    best_lamb = comparison[ind_best,0]
    accu = -comparison[ind_best,1]
   
    return best_lamb, accu

def cross_validation(y, x, fun, k_indices, k, lamb , log=False ):
    """
    Completes k-fold cross-validation for Least Squares with GD, SGD, Normal Equations, Logistic and Regularized Logistic 
    Regression with SGD
    """
    # get k'th subgroup in test, others in train
    msk_test = k_indices[k]
    msk_train = np.delete(k_indices, (k), axis=0).ravel()

    x_train = x[msk_train, :]
    x_test = x[msk_test, :]
    y_train = y[msk_train]
    y_test = y[msk_test]

    # initialize output vectors
    y_train_pred = np.zeros(len(y_train))
    y_test_pred = np.zeros(len(y_test))
        
    # compute weights using given method

    weights, _ = fun(y_train, x_train, lamb)
       
    # predict
    y_train_pred = np.dot(x_train, weights)
    y_test_pred = np.dot(x_test,weights)
        

    # compute accuracy for train and test data
    e_train = y_train - y_train_pred
    e_test = y_test - y_test_pred
    mse_train = -e_train.T.dot(e_train) /(2*len(e_train))
    mse_test = -e_test.T.dot(e_test) /(2*len(e_test))
    
    return mse_train, mse_test