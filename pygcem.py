# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 02:08:24 2014

@author: hangyin
"""

# a python implementation of Cross-Entropy method for stochastic optimization
import copy
import numpy as np

class GaussianCEM:
    def __init__(self, x0=[0], eliteness=10, covar_full=False, covar_learning_rate=1, covar_scale=None, covar_bounds=[0.1]):
        self.options = {
                            'dist_weighting_method':            'PI-BB',
                            'dist_cov_update':                  'PI-BB',
                            'dist_eliteness':                   eliteness,
                            'covar_full':                       covar_full,
                            'covar_learning_rate':              covar_learning_rate,
                            'covar_scale':                      covar_scale,
                            'covar_bounds':                     covar_bounds
                        }

        self.mean = np.array(x0)
        self.covar = np.eye(len(self.mean))
        return

    def fit(self, X, y):
        #one fit step given samples and cost values of the queries
        dist_parms = {'mean':self.mean, 'covar':self.covar}
        res_dist_parms, _ = self.default_dist_fitting(dist_parms, X, y, self.options)

        self.mean = res_dist_parms['mean']
        self.covar = res_dist_parms['covar']
        return

    def sample(self, n_samples=10):
        #take gaussian sample from the current internal distribution
        return np.random.multivariate_normal(self.mean, self.covar, n_samples)

    #default fitting, just to fit a gaussian
    def default_dist_fitting(self, dist_parms, samples, costs, options, var_dim=None):
        if var_dim is None:
            var_dim = (0, len(dist_parms['mean']))
        #this is the key function, need to use weighted samples according to
        #evaluated cost to work out updated parameters for the distribution
        #map the cost to weight
        #check relative standard deviation, if small enough, then all samples should be treated equally
        if np.std(costs) / np.mean(costs) < 1e-8:
            weights = np.ones(np.shape(costs))
        else:
            if options['dist_weighting_method'] == 'PI-BB':
                #take exponential of the cost
                h = options['dist_eliteness']
                weights = np.exp(-h*(costs - min(costs)) / (max(costs) - min(costs) + 1e-6))
            elif options['dist_weighting_method'] == 'CEM' \
                or options['dist_weighting_method'] == 'CMA-ES':
                #CEM/CMA-ES heuristics, rank them, mu is the number of samples to select
                mu = options['dist_eliteness']
                sorted_costs_idx = np.argsort(costs)
                weights = np.zeros(np.shape(costs))
                if options['dist_weighting_method'] == 'CEM':
                    weights[sorted_costs_idx[:mu]] = 1 / mu
                else:
                    weights[sorted_costs_idx[:mu]] = np.log(mu + 1./2) - np.log(range(mu) + 1)
            else:
                print "Undefined distribution weighting method {0}.".format(options['dist_weighting_method'])
                return None

        #normalize the weights, estimate weighted mean/covariance, herustics for covariance if necessary
        weights = weights / np.sum(weights)
        #compute weighted mean
        res_dist_parms = copy.deepcopy(dist_parms)
        res_dist_parms['mean'] = np.sum([samples[i] * weights[i] for i in range(len(weights))], axis=0)
        #weighted covariance
        if options['dist_cov_update'] == 'decay':
            #decay the covariance, turns out to be similar to Simulated Annealing
            covar_new = options['dist_cov_decay'] * dist_parms['covar']
        elif options['dist_cov_update'] == 'PI-BB' or options['dist_cov_update'] == 'CEM':
            mu = dist_parms['mean']
            if options['dist_cov_update'] == 'CEM':
                #for standard CEM use the mean of new distribution
                mu = res_dist_parms['mean']
            epsilon = samples - np.tile(mu, [samples.shape[0], 1])
            #print epsilon
            #print weights
            covar_new = np.dot(np.transpose([epsilon[i, :] * weights[i] for i in range(len(weights))]), epsilon)
            #diagonalize if required, a heuristics is that diagonal covariance is relatively stable
            if options['covar_full'] is False:
                covar_new = np.diag(np.diag(covar_new))
            #apply learning rate
            covar_new = (1 - options['covar_learning_rate']) * dist_parms['covar'] + \
                options['covar_learning_rate'] * covar_new
        else:
            print "CMA_ES covariance update has not been implemented yet"
            covar_new = dist_parms['covar']

        if options['covar_bounds'] is None:
            covar_new_bounded = covar_new
        else:
            #allow bounds on covariance matrix to prevent inmature convergence
            #apply scale to dimensions, if not specified, treat them equally
            if options['covar_scale'] is None:
                covar_scale = np.ones(np.shape(res_dist_parms['mean']))
            else:
                covar_scale = options['covar_scale']
            #from 1D-array to 2D array for transpose product
            covar_scale = covar_scale[np.newaxis]
            covar_scaled = covar_new * (covar_scale.T.dot(covar_scale))
            #The first is the relative lower bound
            rel_lower_bound = options['covar_bounds'][0]
            #The second is the absolute lower bound if available
            if len(options['covar_bounds']) > 1:
                abs_lower_bound = options['covar_bounds'][1]
            else:
                #no valid bound
                abs_lower_bound = -1
            #The third is the absolute upper bound to prevent explore too much
            if len(options['covar_bounds']) > 2:
                abs_upper_bound = options['covar_bounds'][2]
            else:
                #no valid bound
                abs_upper_bound = -1
            #extract eigen values to control the descent direction
            #if covar is diagonal, directly manipulate it, avoid expensive eigen decomposition
            if options['covar_full'] is False:
                eig_vec = np.eye(np.shape(covar_scaled)[0])
                eig_val = np.diag(covar_scaled).copy()
            else:
                #need to do eigen decomposition...
                eig_val, eig_vec = np.linalg.eig(covar_scaled)
            #apply bounds to eigen values
            #absolute upper
            if abs_upper_bound != -1:
                eig_val[eig_val > abs_upper_bound] = abs_upper_bound
            #relative lower bound
            if rel_lower_bound != 1:
                if rel_lower_bound < 0 or rel_lower_bound > 1:
                    print "Relative bound should be within [0, 1]"
                else:
                    #calculate a candidate lower bound from relative value
                    abs_lower_bound = max([abs_lower_bound, rel_lower_bound * max(eig_val)])
            #absolute lower
            if abs_lower_bound != -1:
                eig_val[eig_val < abs_lower_bound] = abs_lower_bound
            #reconstruct covariance matrix
            covar_new_bounded = eig_vec.dot(np.diag(eig_val)).dot(eig_vec.T)
            #rescale
            covar_new_bounded = covar_new_bounded / (covar_scale.T.dot(covar_scale))
        res_dist_parms['covar'][var_dim[0]:var_dim[1], var_dim[0]:var_dim[1]] = covar_new_bounded[var_dim[0]:var_dim[1], var_dim[0]:var_dim[1]]
        return res_dist_parms, res_dist_parms['mean']


def pygcem_test():

    n_iters = 100

    #define a task, search a stationary point
    x_dest = np.array([0, 0])
    x_init = np.array([5, 5])

    gcem = GaussianCEM(x0=x_init)
    curr_mean = gcem.mean

    #take goal
    def test_cost(x):
        return np.linalg.norm(x - x_dest, axis=1)

    for i in range(n_iters):
        #take some rollouts from the current policy
        #just use the internal sampling
        X = gcem.sample(n_samples=10)
        y = test_cost(X)
        curr_avg_cost = np.mean(y)
        print 'Mean: ', curr_mean, '; Avg cost: ', curr_avg_cost

        gcem.fit(X, y)
        diff = curr_mean - gcem.mean
        if np.linalg.norm(diff) < 1e-6:
            break
        curr_mean = gcem.mean

    return

if __name__ == '__main__':
    pygcem_test()
