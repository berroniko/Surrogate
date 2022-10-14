import numpy as np
from scipy.stats import qmc


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