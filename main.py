import pickle
import math
import pandas as pd
import pprint

from sklearn import svm
from sklearn.metrics import r2_score
from sklearn import neighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform

from sampler import lhs_sample, random_sample


def select_features(k, X_train, y_train):
    """using SelectKBest"""
    skb = SelectKBest(f_regression, k=k).fit(X=X_train, y=y_train)
    print()
    print("-----------------------------------------------------------------------------")
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
    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    print(f"kernel: {svr.get_params().get('estimator__kernel')}")
    print(f"C: {svr.get_params().get('estimator__C')}")
    print(f"SVR R2    {r2:11.2f}")
    print()
    return svr, r2


def gpr(X_train, y_train, X_test, y_test):
    """regression with Gaussian Process Regressor"""
    # https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)
    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    print(f"gpr R2    {r2:11.2f}")
    print()
    return gpr, r2


def knn(X_train, y_train, X_test, y_test):
    """regression with nearest neighbors"""
    # GridSearch
    param_grid = {
        'n_neighbors': [3, 4, 5, 6, 7, 8, 9],
        'weights': ['distance', 'uniform']}
    knn = GridSearchCV(estimator=neighbors.KNeighborsRegressor(), param_grid=param_grid)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    r2 = r2_score(y_true=y_test, y_pred=y_pred_knn)
    print(f"nr of neighbors: {knn.get_params().get('estimator__n_neighbors')}")
    print(f"weights: {knn.get_params().get('estimator__weights')}")
    print(f"knn R2    {r2:11.2f}")
    print()
    return knn, r2


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

    results = []
    # define the possible range of feature numbers
    nr_feature_options = list(range(2, len(parameters) + 1))

    for k in nr_feature_options:
        # select the best features
        skb = select_features(k=k, X_train=X, y_train=y_train)
        X_train = skb.transform(X=X)
        X_test = skb.transform(X=X_t)

        # train the different regressors
        gpr_reg, r2_gpr = gpr(X_train, y_train, X_test, y_test)
        results.append({"name": "gpr", "nr_features": k, "r2": r2_gpr, "regressor": gpr_reg})
        knn_reg, r2_knn = knn(X_train, y_train, X_test, y_test)
        results.append({"name": "knn", "nr_features": k, "r2": r2_knn, "regressor": knn_reg})
        svr_reg, r2_svr = svr(X_train, y_train, X_test, y_test)
        results.append({"name": "svr", "nr_features": k, "r2": r2_svr, "regressor": svr_reg})

    df = pd.DataFrame(results)

    df_disp = df.drop(["regressor"], axis=1)
    pprint.pprint(df_disp)

    # identify and save the best Regressor
    best = df.query('r2 == r2.max()')

    print()
    print(
        f"Best regressor is {best.iloc[0]['name']} with {best.iloc[0]['nr_features']} features achieving "
        f"an R2 of {best.iloc[0]['r2']:.2}")

    with open("regressor.pkl", "wb") as f:
        pickle.dump(best["regressor"], f)
