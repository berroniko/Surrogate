import pickle
import math

from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import neighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline

from sampler import lhs_sample, random_sample


def select_features(k, X_train, y_train):
    """using SelectKBest"""
    k = 2
    skb = SelectKBest(f_regression, k=k).fit(X=X_train, y=y_train)
    print(f"The {k} best features: {skb.get_feature_names_out(list(parameters.keys()))}")
    print()
    return skb


def svr(X_train, y_train, X_test, y_test):
    """regression using SVR (Support Vector Regression)"""
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': loguniform(1e0, 1e3)}
    svr = RandomizedSearchCV(estimator=svm.SVR(), param_distributions=param_grid)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    print(f"kernel: {svr.get_params().get('estimator__kernel')}")
    print(f"C: {svr.get_params().get('estimator__C')}")
    print(f"SVR R2    {r2_score(y_true=y_test, y_pred=y_pred):11.2f}")
    print()
    return svr, y_pred


def gpr(X_train, y_train, X_test, y_test):
    """regression with Gaussian Process Regressor"""
    # https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)
    print(f"gpr R2    {r2_score(y_true=y_test, y_pred=y_pred):11.2f}")
    print()
    return gpr, y_pred


def knn(X_train, y_train, X_test, y_test):
    """regression with nearest neighbors"""
    # GridSearch
    param_grid = {
        'n_neighbors': [3, 4, 5, 6, 7, 8, 9],
        'weights': ['distance', 'uniform']}
    knn = GridSearchCV(estimator=neighbors.KNeighborsRegressor(), param_grid=param_grid)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print(f"nr of neighbors: {knn.get_params().get('estimator__n_neighbors')}")
    print(f"weights: {knn.get_params().get('estimator__weights')}")
    print(f"knn R2    {r2_score(y_true=y_test, y_pred=y_pred_knn):11.2f}")
    print()
    return knn, y_pred_knn


if __name__ == '__main__':
    # define parameter names, lower and upper bound
    parameters = {
        "x1": (0, 10),
        "x2": (-1, 5),
        "x3": (200, 300),
        "x4": (-math.pi, math.pi),
    }


    def origin_function(x1, x2, x3, x4):
        return x1 ** 3 + (x2 ** 2) * math.sin(x4) + x3 * 2


    # Generate training and test data
    X = lhs_sample(nr_sample_points=1000, parameter_dict=parameters)
    # X = linear_sample(500, parameter_dict=parameters)
    X_t = random_sample(nr_sample_points=500, parameter_dict=parameters)
    y_train = [origin_function(*i) for i in X]
    y_test = [origin_function(*i) for i in X_t]

    # select the best features
    k = 2
    skb = select_features(k=k, X_train=X, y_train=y_train)
    X_train = skb.transform(X=X)
    X_test = skb.transform(X=X_t)

    gpr(X_train, y_train, X_test, y_test)
    knn(X_train, y_train, X_test, y_test)
    svr(X_train, y_train, X_test, y_test)

    # save the Regressor
    with open("regressor.pkl", "wb") as f:
        pickle.dump(knn, f)
