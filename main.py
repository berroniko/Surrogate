from scipy.stats import qmc
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

# define parameter names, lower and upper bound
parameters = {
    "x1": (0, 10),
    "x2": (-1, 5),
    "x3": (200, 1000)
}


def linear_sample(nr_sample_points, parameter_dict):
    """
    Linear Samples evenly spaced within lower and upper bound of each parameter
    :param nr_sample_points: number of sample points to be returned
    :param parameter_dict: dictionary of parameter_name: (min, max)
    # :return: dictionary, key: parameter name, value: list of evenly spaced samples points
    :return: list of parameter lists
    """
    samples = [np.linspace(v[0], v[1], nr_sample_points) for v in parameter_dict.values()]
    return list(zip(*samples))  # transpose
    # return {k: np.linspace(v[0], v[1], nr_sample_points) for k, v in parameter_dict.items()}


def lhs_sample(nr_sample_points, parameter_dict):
    """
    Latin Hypercube Samples within lower and upper bound of each parameter
    :param nr_sample_points: number of sample points to be returned
    :param parameter_dict: dictionary of parameter_name: (min, max)
    :return: list of parameter lists
    """
    dimension = len(parameter_dict)

    sampler = qmc.LatinHypercube(d=dimension)
    sample = sampler.random(n=nr_sample_points)
    l_bounds = [i[0] for i in parameter_dict.values()]
    u_bounds = [i[1] for i in parameter_dict.values()]
    return qmc.scale(sample, l_bounds, u_bounds)


def random_sample(nr_sample_points, parameter_dict):
    """
    Random Samples within lower and upper bound of each parameter
    :param nr_sample_points: number of sample points to be returned
    :param parameter_dict: dictionary of parameter_name: (min, max)
    :return: list of parameter lists
    """
    samples = [np.random.uniform(v[0], v[1], nr_sample_points) for v in parameter_dict.values()]
    return list(zip(*samples))  # transpose


def origin_function(x1, x2, x3):
    return x1 + x2 ** 2 + x3 * 2


X = lhs_sample(500, parameter_dict=parameters)
X_test = random_sample(200, parameter_dict=parameters)
# X = linear_sample(500, parameter_dict=parameters)
y_true = [origin_function(*i) for i in X]
y_test = [origin_function(*i) for i in X_test]

# regression using SVR
regr = svm.SVR(kernel="rbf")
regr.fit(X, y_true)
y_pred = regr.predict(X_test)
print(f"SVR error {mean_squared_error(y_true=y_test, y_pred=y_pred):9.1f}")

# regression with Gaussian Process Regressor
# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
gpr.fit(X, y_true)
y_pred_gpr = gpr.predict(X_test)
print(f"gpr error {mean_squared_error(y_true=y_test, y_pred=y_pred_gpr):9.1f}")

# regression with nearest neighbors
knn = neighbors.KNeighborsRegressor(n_neighbors=5, weights="distance")
knn.fit(X, y_true)
y_pred_knn = knn.predict(X_test)
print(f"knn error {mean_squared_error(y_true=y_test, y_pred=y_pred_knn):9.1f}")


# TODO pipelines for parameter optimization
# TODO dimension reduction

