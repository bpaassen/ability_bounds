"""
Classes to fit item response theory (IRT) models to data, including
lower and upper bounds for ability parameters.

"""

# Faster Confidence Intervals for Item Response Theory
# via an Approximate Likelihood Profile
# Copyright (C) 2021-2022
# Benjamin Paaßen
# German Research Center for Artificial Intelligence
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2021-2022, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '0.1.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@dfki.de'

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from scipy.stats import norm
from scipy.stats import chi2



class IRT:
    """ An implementation of basic IRT from which we can inherit.

    Attributes
    ----------
    regul: float (default = 1.)
        The L2 regularization strength C when fitting the IRT model.
    theta_: ndarray
        The learned ability parameters.
    b_: ndarray
        The learned difficulties.

    """
    def __init__(self, regul = 1.):
        self.regul = regul

    def fit(self, X):
        m, n = X.shape

        # extract false and correct answers
        Pos  = X > 0.5
        Neg  = X < 0.5
        Nan  = np.isnan(X)
        # set up IRT objective function for the current dataset
        def irt_obj(theta, b):
            # compute predicted probability
            Z      = np.expand_dims(theta, 1) - np.expand_dims(b, 0)
            P      = np.zeros_like(Z)
            hi     = Z > 10.
            P[hi]  = 1.
            mid    = np.abs(Z) <= 10.
            P[mid] = 1. / (1. + np.exp(-Z[mid]))
            # compute gradients
            Delta  = P - X
            Delta[Nan] = 0.
            thetagrad  =  np.sum(Delta, 1) + self.regul * theta
            bgrad      = -np.sum(Delta, 0) + self.regul * b
            # check for degenerate cases, where the log would return infinity
            if np.any(P[Pos] < 1E-8) or np.any(1. - P[Neg] < 1E-8):
                return np.inf, thetagrad, bgrad
            # compute loss
            l = -np.sum(np.log(P[Pos]))-np.sum(np.log(1. - P[Neg])) + self.regul * 0.5 * (np.sum(np.square(theta)) + np.sum(np.square(b)))
            # return
            return l, thetagrad, bgrad
        # save it as property of this class
        self.irt_obj_ = irt_obj

        # set up an interface for a single parameter vector
        def obj(params):
            theta = params[:m]
            b     = params[m:]
            l, thetagrad, bgrad = irt_obj(theta, b)
            return l, np.concatenate((thetagrad, bgrad))

        # call optimize
        res = minimize(obj, np.zeros(m+n), jac = True)
        # retrieve result
        self.theta_    = res.x[:m]
        self.b_        = res.x[m:]
        self.res_      = res

        # return
        return self

    def predict(self, X = None):
        Z = np.expand_dims(self.theta_, 1) - np.expand_dims(self.b_, 0)
        Z[Z < 0.] = 0.
        Z[Z > 0.] = 1.
        return Z

    def predict_proba(self, X = None):
        Z = np.expand_dims(self.theta_, 1) - np.expand_dims(self.b_, 0)
        return 1. / (1. + np.exp(-Z))



class BarrierBounds(IRT):
    """ An implementation of our proposed scheme to identify
    ability bounds. We try to find bounds which increase the
    negative log likelihood by at most a factor of (1 + delta).
    To ensure this, we use a log-barrier method with a barrier
    factor of mu.

    Attributes
    ----------
    regul: float (default = 1.)
        The L2 regularization strength C when fitting the IRT model.
    delta: float (default = 0.01)
        The factor by which we are allowed to exceed the minimum
        negative log likelihood.
    absolute_bound: float (default = None)
        If an absolute bound is desired instead of a relative bound.
        This is mostly relevant to remain comparable to the Likelihood
        Profile method.
    mu: float (default = 0.01)
        The factor for the log barrier term in the loss.
    theta_: ndarray
        The learned ability parameters.
    b_: ndarray
        The learned difficulties.
    theta_min_: ndarray
        The lower bounds for theta.
    theta_max_: ndarray
        The upper bounds for theta.

    """
    def __init__(self, regul = 1., delta = 0.01, absolute_bound = None, mu = 0.01):
        super(BarrierBounds, self).__init__(regul)
        self.delta = delta
        self.absolute_bound = absolute_bound
        self.mu    = mu

    def fit(self, X):
        # fit the standard IRT model
        super(BarrierBounds, self).fit(X)

        m, n = X.shape

        # generate mu to ensure that the bounds will be valid
        mu = self.mu * min(1, self.delta * self.res_.fun / np.max(np.abs(self.res_.jac[:m])))

        # set up loss bound
        if self.absolute_bound is None:
            loss_bound = (1. + self.delta) * self.res_.fun
        else:
            loss_bound = self.res_.fun + self.absolute_bound
        # set up initial parameters
        init_params = np.concatenate((self.theta_, self.b_))

        # start identifying bounds for the ability parameters
        self.theta_min_ = np.zeros(m)
        self.theta_max_ = np.zeros(m)
        for i in range(m):
            # construct objective for lower bound
            def obj(params):
                theta = params[:m]
                b     = params[m:]
                # compute regular gradient
                l, thetagrad, bgrad = self.irt_obj_(theta, b)
                # check loss delta for a degenerate case where the log would be infinite
                loss_delta = loss_bound - l
                if loss_delta <= 0.:
                    return np.inf, np.concatenate((thetagrad, bgrad))
                # compute loss for the current problem
                loss = +theta[i] + mu * np.log(loss_delta)
                # compute gradients
                thetagrad  = mu / loss_delta * thetagrad
                thetagrad[i] += 1.
                bgrad      = mu / loss_delta * bgrad
                # return
                return loss, np.concatenate((thetagrad, bgrad))

            # call optimize
            res = minimize(obj, init_params, method = 'L-BFGS-B', jac = True, options = {'maxls' : 100 })
            # retrieve result
            self.theta_min_[i] = res.x[i]

            # construct objective for upper bound
            def obj(params):
                theta = params[:m]
                b     = params[m:]
                # compute regular gradient
                l, thetagrad, bgrad = self.irt_obj_(theta, b)
                # check loss delta for a degenerate case where the log would be infinite
                loss_delta = loss_bound - l
                if loss_delta <= 0.:
                    return np.inf, np.concatenate((thetagrad, bgrad))
                # compute loss for the current problem
                loss = -theta[i] + mu * np.log(loss_delta)
                # compute gradients
                thetagrad  = mu / loss_delta * thetagrad
                thetagrad[i] -= 1.
                bgrad      = mu / loss_delta * bgrad
                # return
                return loss, np.concatenate((thetagrad, bgrad))

            res = minimize(obj, init_params, method = 'L-BFGS-B', jac = True, options = {'maxls' : 100 })
            # retrieve result
            self.theta_max_[i] = res.x[i]

        return self




class AOBounds(IRT):
    """ An implementation of our proposed scheme to identify
    ability bounds. We try to find bounds which increase the
    negative log likelihood by at most a factor of (1 + delta).
    To ensure this, we use a special solver for alternating 
    optimization where we only change theta[i] and then all other
    parameters.

    Attributes
    ----------
    regul: float (default = 1.)
        The L2 regularization strength C when fitting the IRT model.
    delta: float (default = 0.01)
        The factor by which we are allowed to exceed the minimum
        negative log likelihood.
    absolute_bound: float (default = None)
        If an absolute bound is desired instead of a relative bound.
        This is mostly relevant to remain comparable to the Likelihood
        Profile method.
    num_iterations: int (default = 4)
        The number of binary search iterations to look for the
        bounds.
    theta_: ndarray
        The learned ability parameters.
    b_: ndarray
        The learned difficulties.
    theta_min_: ndarray
        The lower bounds for theta.
    theta_max_: ndarray
        The upper bounds for theta.

    """
    def __init__(self, regul = 1., delta = 0.01, absolute_bound = None, num_iterations = 4):
        super(AOBounds, self).__init__(regul)
        self.delta = delta
        self.absolute_bound = absolute_bound
        self.num_iterations = num_iterations


    def find_root_with_ao_(self, X, Pos, Neg, Nan, i, theta_i_init, li, lnoti, loss_bound):
        """ This function implements a custom root solver which
        ensures that lnoti + li becomes almost exactly equal to the
        loss bound.

        """
        m, n = X.shape

        theta = self.theta_
        b     = self.b_
        theta_i = theta_i_init
        for t in range(self.num_iterations):
            # compute the current loss that is due to theta[i]
            # but only in iterations larger zero because the initial
            # value is already given in the parent function
            if t > 0:
                # compute predicted probability
                Z      = np.expand_dims(theta, 1) - np.expand_dims(b, 0)
                P      = np.zeros_like(Z)
                hi     = Z > 10.
                P[hi]  = 1.
                mid    = np.abs(Z) <= 10.
                P[mid] = 1. / (1. + np.exp(-Z[mid]))
                # compute loss
                l = -np.sum(np.log(P[Pos]))-np.sum(np.log(1. - P[Neg])) + self.regul * 0.5 * (np.sum(np.square(theta)) + np.sum(np.square(b)))
                # compute loss for i only
                li = -np.sum(np.log(P[i, Pos[i, :]]))-np.sum(np.log(1. - P[i, Neg[i, :]])) + self.regul * 0.5 * theta[i] ** 2
                lnoti = l - li

            # find the theta_i value that yields exactly the loss
            # bound, provided that we don't change the other parameters.
            # This will be an overestimate of the actual loss we can
            # achieve for this theta_i value

            # set up the objective function for root finding
            def root_obj(theta_i):
                # compute predicted probability
                z      = theta_i - b
                p      = np.zeros_like(z)
                hi     = z > 10.
                p[hi]  = 1.
                mid    = np.abs(z) <= 10.
                p[mid] = 1. / (1. + np.exp(-z[mid]))
                # compute gradients
                delta  = p - X[i, :]
                delta[Nan[i, :]] = 0.
                thetagrad  =  np.sum(delta) + self.regul * theta_i
                # check for degenerate cases, where the log would return infinity
                if np.any(p[Pos[i, :]] < 1E-8) or np.any(1. - p[Neg[i, :]] < 1E-8):
                    return np.inf, thetagrad
                # compute loss
                li = -np.sum(np.log(p[Pos[i, :]]))-np.sum(np.log(1. - p[Neg[i, :]])) + self.regul * 0.5 * theta_i ** 2
                l  = lnoti + li
                # return
                return l - loss_bound, thetagrad

            # start root finding process
            res = root_scalar(root_obj, x0 = theta_i, fprime = True)
            # the root is our new theta_i value
            theta_i = res.root
            # avoid another costly L-BFGS-B optimization if we
            # are in the last iteration
            if t == self.num_iterations - 1:
                break
            # optimize the negative log likelihood with all other
            # parameters except theta[i]
            def obj(params):
                theta = params[:m]
                theta[i] = theta_i
                b     = params[m:]
                # compute regular gradient
                l, thetagrad, bgrad = self.irt_obj_(theta, b)
                thetagrad[i] = 0.
                return l, np.concatenate((thetagrad, bgrad))
            # call optimize
            res = minimize(obj, np.concatenate((theta, b)), method = 'L-BFGS-B', jac = True, options = {'maxls' : 100 })
            theta = res.x[:m]
            theta[i] = theta_i
            b     = res.x[m:]
        return theta_i


    def fit(self, X):
        # fit the standard IRT model
        super(AOBounds, self).fit(X)

        m, n = X.shape

        # set up loss bound
        if self.absolute_bound is None:
            loss_bound = (1. + self.delta) * self.res_.fun
        else:
            loss_bound = self.res_.fun + self.absolute_bound

        # compute the loss separately for all i
        Pos  = X > 0.5
        Neg  = X < 0.5
        Nan  = np.isnan(X)
        # compute predicted probability
        Z      = np.expand_dims(self.theta_, 1) - np.expand_dims(self.b_, 0)
        P      = np.zeros_like(Z)
        hi     = Z > 10.
        P[hi]  = 1.
        mid    = np.abs(Z) <= 10.
        P[mid] = 1. / (1. + np.exp(-Z[mid]))
        # compute loss
        L = np.zeros(m)
        for i in range(m):
            L[i] = -np.sum(np.log(P[i, Pos[i, :]]))-np.sum(np.log(1. - P[i, Neg[i, :]]))
        L = L + self.regul * 0.5 * np.square(self.theta_)

        # start identifying bounds for the ability parameters
        self.theta_min_ = np.zeros(m)
        self.theta_max_ = np.zeros(m)
        for i in range(m):
            # start search for lower bound.
            self.theta_min_[i] = self.find_root_with_ao_(X, Pos, Neg, Nan, i, self.theta_[i] - 1, L[i], self.res_.fun - L[i], loss_bound)
            # start search for upper bound.
            self.theta_max_[i] = self.find_root_with_ao_(X, Pos, Neg, Nan, i, self.theta_[i] + 1, L[i], self.res_.fun - L[i], loss_bound)

        return self




class LikelihoodProfile(IRT):
    """ This is an implementation of the Likelihood profile
    estimation method proposed by Chalmers, Pek, & Liu (2017).
    doi:10.1080/00273171.2017.1329082.

    This method first trains a vanilla IRT model and then
    performs a separate optimization for each ability parameter,
    where we look for the value which is as small/large as possible
    such that a likelihood ratio test would not reject the null
    hypothesis at the desired confidence level.
    This is equivalent to minimizing/maximizing the parameter
    value such that twice the difference between the log likelihood
    and the optimal log likelihood is equal to the chi-square
    quantile for the desired confidence level.

    Attributes
    ----------
    regul: float (default = 1.)
        The L2 regularization strength C when fitting the IRT model.
    alpha: float (default = .95)
        The desired confidence level.
    theta_: ndarray
        The learned ability parameters.
    b_: ndarray
        The learned difficulties.
    theta_min_: ndarray
        The lower bounds for theta.
    theta_max_: ndarray
        The upper bounds for theta.

    """
    def __init__(self, regul = 1., alpha = .95):
        super(LikelihoodProfile, self).__init__(regul)
        self.alpha = alpha

    def fit(self, X):
        # fit the standard IRT model
        super(LikelihoodProfile, self).fit(X)

        m, n = X.shape

        # set up an upper bound for the loss via the chi square
        # quantile function
        bound = 2*self.res_.fun + chi2.ppf(self.alpha, df = 1)

        # set up initial parameters
        init_params = np.concatenate((self.theta_, self.b_))

        # start identifying bounds for the ability parameters
        self.theta_min_ = np.zeros(m)
        self.theta_max_ = np.zeros(m)
        for i in range(m):
            # construct objective for root finding
            def root_obj(theta_i):
                # construct inner objective for maximizing the
                # likelihood at fixed theta_i
                def obj(params):
                    theta = params[:m]
                    theta[i] = theta_i
                    b     = params[m:]
                    # compute regular gradient
                    l, thetagrad, bgrad = self.irt_obj_(theta, b)
                    thetagrad[i] = 0.
                    return l, np.concatenate((thetagrad, bgrad))
                # perform inner optimization
                res = minimize(obj, init_params, method = 'L-BFGS-B', jac = True, options = {'maxls' : 100 })
                # compute loss for root finding
                root_loss = 2*res.fun - bound
                # compute derivative for root finding
                theta = res.x[:m]
                theta[i] = theta_i
                b = res.x[m:]
                _, thetagrad, _ = self.irt_obj_(theta, b)
                return root_loss, 2*thetagrad[i]
            # perform root finding process for lower bound, starting
            # well above the optimal value
            res = root_scalar(root_obj, x0 = self.theta_[i] - 1, fprime = True)
            self.theta_min_[i] = res.root
            # perform root finding process for upper bound, starting
            # well above the optimal value
            res = root_scalar(root_obj, x0 = self.theta_[i] + 1, fprime = True)
            self.theta_max_[i] = res.root

        return self


class WaldBounds(IRT):
    """ This is an implementation of the Wald method for
    confidence bounds as explained in Chalmers, Pek, & Liu (2017).
    doi:10.1080/00273171.2017.1329082.

    This method first trains a vanilla IRT model and then
    uses the L-BFGS estimate for the inverse Hessian to estimate
    where the negative log likelihood would exceed a certain bound.

    Attributes
    ----------
    regul: float (default = 1.)
        The L2 regularization strength C when fitting the IRT model.
    alpha: float (default = .95)
        The desired confidence level.
    theta_: ndarray
        The learned ability parameters.
    b_: ndarray
        The learned difficulties.
    theta_min_: ndarray
        The lower bounds for theta.
    theta_max_: ndarray
        The upper bounds for theta.

    """
    def __init__(self, regul = 1., alpha = .95):
        super(WaldBounds, self).__init__(regul)
        self.alpha = alpha

    def fit(self, X):
        # fit the standard IRT model
        super(WaldBounds, self).fit(X)

        m, n = X.shape

        # infer standard deviations for parameters based on the
        # square root of the diagonal of the inverse Hessian
        stds = np.sqrt(np.diag(self.res_.hess_inv)[:m])

        # multiply standard deviations with Gaussian quantile
        scalar = norm.ppf(1. - .5 * (1. - self.alpha))
        self.theta_min_ = self.theta_ - scalar * stds
        self.theta_max_ = self.theta_ + scalar * stds

        return self

