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
from sklearn.pipeline import Pipeline, make_pipeline

from sampler import lhs_sample, random_sample

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
X_train = lhs_sample(1000, parameter_dict=parameters)
# X = linear_sample(500, parameter_dict=parameters)
X_test = random_sample(500, parameter_dict=parameters)
y_train = [origin_function(*i) for i in X_train]
y_test = [origin_function(*i) for i in X_test]



# Automatic regression selection and parameter optimization using Pipeline and GridSearchCV

# for regression with Gaussian Process Regressor see:
# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor

pipe = Pipeline(
    [
        ("select_feature", "-"),
        ("classify", "-"),
    ]
)

# possible nr of features:
nr_feature_options = list(range(2, len(parameters)+1))

param_grid = [
    {
        "select_feature": [SelectKBest(score_func=f_regression)],
        "select_feature__k": nr_feature_options,
        "classify": [GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(), random_state=0)]
    },
    {
        "select_feature": [SelectKBest(score_func=f_regression)],
        "select_feature__k": nr_feature_options,
        "classify": [neighbors.KNeighborsRegressor()],
        "classify__n_neighbors": [3, 4, 5, 6, 7, 8, 9],
        "classify__weights": ['distance', 'uniform']
    },
    {
        "select_feature": [SelectKBest(score_func=f_regression)],
        "select_feature__k": nr_feature_options,
        "classify": [svm.SVR()],
        "classify__C": [10e-2, 10e-1, 10e0, 10e1, 10e2],
        "classify__kernel": ['linear', 'poly', 'rbf', 'sigmoid']
    }
]

grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid, scoring='r2')
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)

print(f"{grid.best_estimator_ = }")
print(f"{grid.best_params_ = }")
print(f"grid R2    {r2_score(y_true=y_test, y_pred=y_pred):11.2f}")
print()


# save the Regressor
with open("regressor.pkl", "wb") as f:
    pickle.dump(grid, f)

