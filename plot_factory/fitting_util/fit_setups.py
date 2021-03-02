from iminuit import Minuit
import numpy as np

from .binomial_error import get_binomial_1sigma_simplified


def gaussian_minimization(data, n_data, set_limits=False, binomial_error=False):
    from iminuit.cost import LeastSquares
    M = data.M
    Y = data.Y / data.Y.sum()
    if binomial_error:
        y_err = get_binomial_1sigma_simplified(Y*n_data)
    else:
        y_err = (Y / n_data)**.5 # == (Y*n_data)**.5 / n_data
    def fct(mc_matrix, x):
        return mc_matrix.dot(x)
    def fct_model_counts_per_bin(x):
        return M.dot(x)
    least_squares = LeastSquares(M, Y, y_err, fct)
    minimizer = Minuit(least_squares, data.X0, name=data.x_names)
    minimizer.errordef = Minuit.LEAST_SQUARES
    if set_limits:
        minimizer.limits = (0, 1)
    minimizer.migrad(ncall=10_000)
    return minimizer, fct_model_counts_per_bin


def gaussian_minimization_with_limits(data, n_data):
    return gaussian_minimization(data, n_data, set_limits=True)

def gaussian_minimization_binomial_error(data, n_data):
    return gaussian_minimization(data, n_data, binomial_error=True)

def gaussian_minimization_with_limits_binomial_error(data, n_data):
    return gaussian_minimization(data, n_data, set_limits=True, binomial_error=True)


def binomial_minimization(data, n_data, set_limits=False):
    Y = data.Y / data.Y.sum()
    # dropped_idx = data.X0.argmax()
    dropped_idx = -2  # -> H->ZZ*.
    X0_B = np.delete(data.X0, dropped_idx)
    x_names_B = np.delete(data.x_names, dropped_idx)
    M_B = np.delete(data.M, dropped_idx, axis=1)
    M_B_constraint = data.M[:, dropped_idx]
    def fct_model_counts_per_bin(x):
        return M_B.dot(x) + M_B_constraint.dot(1 - x.sum(axis=-1))
    def binomial_cost_fct(x):
        return - n_data * Y.dot(np.log(fct_model_counts_per_bin(x)))
    minimizer = Minuit(binomial_cost_fct, X0_B, name=x_names_B)
    if set_limits:
        minimizer.limits = (0, 1)
    minimizer.errordef = Minuit.LIKELIHOOD
    minimizer.migrad(ncall=10_000)
    return minimizer, fct_model_counts_per_bin


def binomial_minimization_with_limits(data, n_data):
    return binomial_minimization(data, n_data, set_limits=True)


def poisson_minimization(data, n_data):
    M = data.M
    Y = data.Y / data.Y.sum()
    X0 = n_data * data.X0
    def fct_model_counts_per_bin(x):
        return M.dot(x)
    def fct_model_counts_per_bin_normalized(x):
        y = fct_model_counts_per_bin(x)
        return y / sum(y)
    def poisson_cost_fct(x):
        nu = fct_model_counts_per_bin(x)
        return - n_data * Y.dot(np.log(nu)) + nu.sum()
    minimizer = Minuit(poisson_cost_fct, X0, name=data.x_names)
    minimizer.errordef = Minuit.LIKELIHOOD
    minimizer.migrad(ncall=10_000)
    return minimizer, fct_model_counts_per_bin_normalized
