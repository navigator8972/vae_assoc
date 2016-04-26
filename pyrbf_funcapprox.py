import numpy as np
import matplotlib.pyplot as plt

class PyRBF_FunctionApproximator():
    """
    an RBF function approximator for mono input and mono output...
    note the features are a series of RBF basis function and a constant offset term
    this is used to make the model can be easily initialized as a linear model...
    """
    def __init__(self, rbf_type='gaussian', K=9, normalize=True):

        self.K_ = K
        self.type_ = rbf_type
        self.rbf_parms_ = dict()
        self.prepare_rbf_parameters()

        self.theta_ = np.concatenate([np.zeros(self.K_), [0]])
        self.normalize_rbf_ = normalize
        self.upper_limit_ = None
        self.lower_limit_ = None

        #a function to map parameter theta to the linear constrainted space...
        self.apply_lin_cons = None
        return

    def prepare_rbf_parameters(self):
        #prepare rbf parameters
        #gaussian
        if self.type_ == 'gaussian':
            self.rbf_parms_['mu'] = np.linspace(0.1, 0.9, self.K_)
            self.rbf_parms_['sigma'] = 1. / self.K_
        elif self.type_ == 'sigmoid':
            #logistic curve, there might be other alternatives: e.g., erf, tanh
            self.rbf_parms_['tau'] = self.K_ * 2
            self.rbf_parms_['t0'] = np.linspace(1./self.K_, 1.0, self.K_)
        else:
            print 'Unknown RBF type'

        return

    def set_linear_equ_constraints(self, phases, target=None):
        """
        this funciton allows to set linear equality constraints with the form
        \Phi(phases)^T \theta == target
        target is zero vector if not specified...
        """
        if target is None:
            const_rhs = np.zeros(len(phases))
        else:
            const_rhs = target

        if len(const_rhs) == len(phases):
            #valid constraint
            #evaluate features at constrainted phase points
            self.cons_feats = self.get_features(phases)
            self.cons_invmat = np.linalg.pinv(self.cons_feats.T.dot(self.cons_feats))
            self.cons_offset = const_rhs

            self.apply_lin_cons = lambda theta_old: theta_old - self.cons_feats.dot(
                self.cons_invmat.dot(self.cons_feats.T.dot(theta_old) - self.cons_offset))
        return

    def set_theta(self, theta):
        self.theta_ = theta
        return

    def set_const_offset(self, offset):
        #the last theta...
        self.theta_[-1] = offset
        return

    def rbf_gaussian_evaluation(self, z, mu, sigma):
        res = np.exp(-(z-mu)**2/sigma)
        return res

    def rbf_sigmoid_evaluation(self, z, t0, tau):
        res = 1. / (1 + np.exp(-tau*(z - t0)))
        return res

    def set_limit(self, upper_limit=None, lower_limit=None):
        if upper_limit is not None:
            self.upper_limit_ = upper_limit

        if lower_limit is not None:
            self.lower_limit_ = lower_limit

        return
    def get_features(self, z):
        def get_features_internal(z_var):
            if self.type_ == 'gaussian':
                res = np.array([ self.rbf_gaussian_evaluation(z_var, self.rbf_parms_['mu'][i], self.rbf_parms_['sigma']) for i in range(self.K_)])
                if self.normalize_rbf_:
                    res = res / np.sum(res)
                return np.concatenate([res, [1]])
            elif self.type_ == 'sigmoid':
                res = np.array([ self.rbf_sigmoid_evaluation(z_var, self.rbf_parms_['t0'][i], self.rbf_parms_['tau']) for i in range(self.K_)])
                return np.concatenate([res, [1]])
            else:
                print 'Unknown RBF type'

        res = [get_features_internal(z_var) for z_var in z]
        return np.array(res).T

    def fit(self, z, y, replace_theta=False):
        """
        z: a series of phase variables...
        y: function evaluation
        """
        features = self.get_features(z)

        U, s, V = np.linalg.svd(features.T)
        significant_dims = len(np.where(s>1e-6)[0])

        inv_feats = V.T[:, 0:significant_dims].dot(np.diag(1./s[0:significant_dims])).dot(U[:, 0:significant_dims].T)
        res_theta = inv_feats.dot(y)

        if replace_theta:
            # print 'use fit parameters'
            self.theta_ = res_theta

        return res_theta

    def evaluate(self, z, theta=None, trunc_limit=True):
        """
        evaluate with given phase variables
        """
        features = self.get_features(z)
        if theta is None:
            #use model parameters
            res = features.T.dot(self.theta_)
        else:
            #use given parameters
            res = features.T.dot(theta)

        #truncate with limit if desired
        if trunc_limit:
            #are limits valid?
            if self.upper_limit_ is not None and self.lower_limit_ is not None:
                if self.upper_limit_ > self.lower_limit_:
                    res[res > self.upper_limit_] = self.upper_limit_
                    res[res < self.lower_limit_] = self.lower_limit_

        return res

    def gaussian_sampling(self, theta=None, noise=None, n_samples=10):
        '''
        conducting local gaussian sampling with the given mean theta and noise
        use the current theta is the mean theta is None
        use unit noise if covariance matrix is not given
        '''
        if theta is None:
            mean = self.theta_
        else:
            mean = theta

        if noise is None:
            covar = np.eye(len(self.theta_))
        elif isinstance(noise, int) or isinstance(noise, float):
            covar = np.eye(len(self.theta_)) * noise
        else:
            covar = noise

        #make white gaussian because we might need to apply the linear constraints...
        #<hyin/Feb-07th-2016> hmm, actually this is shifted noise, so remember not to apply that again
        samples = np.random.multivariate_normal(mean, covar, n_samples)
        if self.apply_lin_cons is None:
            res = samples
        else:
            #apply linear constraint to apply the null-space perturbation
            res = [self.apply_lin_cons(s) for s in samples]
        return np.array(res)


def PyRBF_FuncApprox_Test():
    #test
    #fit sin
    n_samples = 100
    z = np.linspace(0.0, 1.0, 100)
    y = np.cos(2*np.pi*z)
    #feature parms
    mu = np.arange(0.1, 1.0, 0.1)
    sigma = 1./len(mu)
    #model
    rbf_mdl = PyRBF_FunctionApproximator(rbf_type='sigmoid', K=10, normalize=True)
    #fit
    res_theta = rbf_mdl.fit(z, y, True)
    print 'fit parameters:', res_theta
    y_hat = rbf_mdl.evaluate(z[n_samples/4:3*n_samples/4])

    #draw the results
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(True)
    ax.plot(z, y, linewidth=3.0)
    ax.plot(z[n_samples/4:3*n_samples/4], y_hat, '.-', linewidth=3.0)
    plt.draw()

    #test for sampling and apply linear constrains
    raw_input('Press ENTER to continue the test of random sampling')
    rbf_mdl = PyRBF_FunctionApproximator(rbf_type='gaussian', K=10, normalize=True)
    y = np.sin(np.linspace(0.0, np.pi, len(z)))
    res_theta = rbf_mdl.fit(z, y, True)
    print 'fit parameters:', res_theta
    #anchoring the initial point...
    rbf_mdl.set_linear_equ_constraints([z[0]], [y[0]])
    #sampling...
    init_fix_samples = rbf_mdl.gaussian_sampling()
    init_fix_trajs = [rbf_mdl.evaluate(z, s) for s in init_fix_samples]
    #anchoring both end points...
    rbf_mdl.set_linear_equ_constraints([z[0], z[-1]], [y[0], y[-1]])
    both_fix_samples = rbf_mdl.gaussian_sampling()
    both_fix_trajs = [rbf_mdl.evaluate(z, s) for s in both_fix_samples]

    print init_fix_samples, both_fix_samples
    #show them...
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.hold(True)
    for traj in init_fix_trajs:
        ax1.plot(z, traj, linewidth=3.0)
    plt.draw()

    ax2 = fig.add_subplot(212)
    ax2.hold(True)
    for traj in both_fix_trajs:
        ax2.plot(z, traj, linewidth=3.0)
    plt.draw()

    return
