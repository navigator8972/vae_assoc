'''
VAE that associate latent representations for different modals of sensory data
'''
import os
import time
import datetime
import numpy as np
from itertools import combinations
import tensorflow as tf

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

class AssocVariationalAutoEncoder(object):
    '''
    Extend Variational AutoEncoder to associate latent representations for two modal of sensory data

    Generally we force the latent variable for two sensory modals to be same by penalizing the divergence of Gaussian distributions
    '''
    def __init__(self, network_architectures, binary=True, transfer_fct=tf.nn.softplus, weights = 1.0, assoc_lambda = 1.0,
                 learning_rate=0.001, batch_size=100):
        self.network_architectures = network_architectures
        self.assoc_lambda = assoc_lambda
        #check if binary data
        if type(binary) is list:
            assert len(binary) == len(network_architectures)
            self.binary = binary
        else:
            self.binary = [binary] * len(network_architectures)

        if type(weights) is list:
            assert len(weights) == len(network_architectures)
            self.weights = weights
        else:
            self.weights = [weights] * len(network_architectures)

        # if type(transfer_fct) is list:
        #     assert len(transfer_fct) == len(network_architectures)
        #     self.transfer_fct = transfer_fct
        # else:
        #     self.transfer_fct = [transfer_fct] * len(network_architectures)
        self.transfer_fct = transfer_fct

        self.learning_rate = learning_rate
        self.batch_size = batch_size


        self.x = [tf.placeholder(tf.float32, [None, na["n_input"]]) for na in self.network_architectures]

        # Create autoencoder network
        self._create_network(batch_size=self.batch_size)
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

        #create a saver for future saving/restoring the model
        self.saver = tf.train.Saver()
        return

    def _unpack_network_arguments(self, scope, hidden_conv, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1,  n_hidden_gener_2,
                            n_input, n_z):
        return scope, hidden_conv, n_hidden_recog_1, n_hidden_recog_2, n_hidden_gener_1,  n_hidden_gener_2, n_input, n_z

    def _create_network(self, batch_size):
        # Initialize autoencode network weights and biases
        # self.network_weights = [self._initialize_weights(**na) for na in self.network_architectures]

        self.z_means = []
        self.z_log_sigma_sqs = []
        self.z_array = []
        self.x_reconstr_means = []

        # Draw one sample z from Gaussian distribution
        # this must be uniform for all networks
        self.n_z = self.network_architectures[0]["n_z"]
        tmp_eps = tf.random_normal((batch_size, self.n_z), 0, 1,
                               dtype=tf.float32)

        for sens_idx, na in enumerate(self.network_architectures):

            # Use recognition network to determine mean and
            # (log) variance of Gaussian distribution in latent
            # space
            tmp_z_mean, tmp_z_log_sigma_sq = \
                self._recognition_network(self.x[sens_idx],
                                          na)
            # z = mu + sigma*epsilon
            tmp_z = tf.add(tmp_z_mean,
                            tf.mul(tf.sqrt(tf.exp(tmp_z_log_sigma_sq)), tmp_eps))

            # Use generator to determine mean of
            # Bernoulli distribution of reconstructed input
            tmp_x_reconstr_mean = \
                self._generator_network(tmp_z,
                                        na,
                                        self.binary[sens_idx],
                                        conv_enable=True)

            #store related tf variables
            self.z_means.append(tmp_z_mean)
            self.z_log_sigma_sqs.append(tmp_z_log_sigma_sq)
            self.z_array.append(tmp_z)
            self.x_reconstr_means.append(tmp_x_reconstr_mean)

        return

    def _initialize_weights(self, scope, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1,  n_hidden_gener_2,
                            n_input, n_z):
        #use scope to differentiate variabels for different sensory information
        with tf.variable_scope(scope):
            all_weights = dict()
            all_weights['weights_recog'] = {
                'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
                'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
                'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
                'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
            all_weights['biases_recog'] = {
                'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
                'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
                'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
                'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
            all_weights['weights_gener'] = {
                'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
                'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
                'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
                'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
            all_weights['biases_gener'] = {
                'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
                'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
                'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
                'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    # def _recognition_network(self, x, weights, biases):
    #     # Generate probabilistic encoder (recognition network), which
    #     # maps inputs onto a normal distribution in latent space.
    #     # The transformation is parametrized and can be learned.
    #     layer_1 = self.transfer_fct(tf.add(tf.matmul(x, weights['h1']),
    #                                        biases['b1']))
    #     layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
    #                                        biases['b2']))
    #     z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
    #                     biases['out_mean'])
    #     z_log_sigma_sq = \
    #         tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
    #                biases['out_log_sigma'])
    #     return (z_mean, z_log_sigma_sq)
    def _recognition_network(self, x, network_architecture):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        scope, hidden_conv, n_hidden_recog_1, n_hidden_recog_2, n_hidden_gener_1,  n_hidden_gener_2, n_input, n_z = self._unpack_network_arguments(**network_architecture)
        with tf.variable_scope(scope):
            if hidden_conv:
                #convolution layer filtering over 2d 28x28 images
                input_size = int(np.sqrt(n_input))
                x_2d = tf.reshape(x, [-1, input_size, input_size, 1])
                #convolution layers use their own initialization of weight variables
                layer_0_5 = conv_2d(    x_2d,
                                        filter_set=[5, 1, n_hidden_recog_1],
                                        stride=2,
                                        padding='SAME',
                                        transfer_fct=None)  #no nonlinear transformation here
                layer_1 = conv_2d(  layer_0_5,
                                    filter_set=[5, n_hidden_recog_1, n_hidden_recog_1*2],
                                    stride=2,
                                    padding='SAME',
                                    transfer_fct=None)
            else:
                weights_hidden_1 = tf.Variable(xavier_init(n_input, n_hidden_recog_1))
                biases_hidden_1 = tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32))
                layer_1 = self.transfer_fct(tf.add(tf.matmul(x, weights_hidden_1),
                                               biases_hidden_1))

            if hidden_conv:
                # layer_1_size = (input_size - 5) / 1
                layer_1_size = input_size / 2 / 2
                #convolution layers use their own initialization of weight variables
                layer_2_2d = conv_2d(   layer_1,
                                        filter_set=[5, n_hidden_recog_1*2, n_hidden_recog_2],
                                        stride=1,
                                        padding='VALID')  #no nonlinear transformation here
                layer_2_size = (layer_1_size - 5) / 1 + 1
                layer_2 = tf.reshape(layer_2_2d, (-1, layer_2_size*layer_2_size*n_hidden_recog_2))
            else:
                weights_hidden_2 = tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2))
                biases_hidden_2 = tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32))
                layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights_hidden_2),
                                               biases_hidden_2))

            if hidden_conv:
                weights_out_mean = tf.Variable(xavier_init(layer_2_size * layer_2_size * n_hidden_recog_2, n_z))
                biases_out_mean = tf.Variable(tf.zeros([n_z], dtype=tf.float32))
                weights_out_log_sigma = tf.Variable(xavier_init(layer_2_size * layer_2_size * n_hidden_recog_2, n_z))
                biases_out_log_sigma = tf.Variable(tf.zeros([n_z], dtype=tf.float32))
            else:
                weights_out_mean = tf.Variable(xavier_init(n_hidden_recog_2, n_z))
                biases_out_mean = tf.Variable(tf.zeros([n_z], dtype=tf.float32))
                weights_out_log_sigma = tf.Variable(xavier_init(n_hidden_recog_2, n_z))
                biases_out_log_sigma = tf.Variable(tf.zeros([n_z], dtype=tf.float32))

            z_mean = tf.add(tf.matmul(layer_2, weights_out_mean),
                            biases_out_mean)
            z_log_sigma_sq = \
                tf.add(tf.matmul(layer_2, weights_out_log_sigma),
                       biases_out_log_sigma)
        return (z_mean, z_log_sigma_sq)

    # def _generator_network(self, z, weights, biases, binary):
    #     # Generate probabilistic decoder (decoder network), which
    #     # maps points in latent space onto a Bernoulli distribution in data space.
    #     # The transformation is parametrized and can be learned.
    #     layer_1 = self.transfer_fct(tf.add(tf.matmul(z, weights['h1']),
    #                                        biases['b1']))
    #     layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
    #                                        biases['b2']))
    #
    #     if binary:
    #         x_reconstr_mean = \
    #             tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
    #                                  biases['out_mean']))
    #     else:
    #         x_reconstr_mean = \
    #             tf.add(tf.matmul(layer_2, weights['out_mean']),
    #                                  biases['out_mean'])
    #     return x_reconstr_mean

    def _generator_network(self, z, network_architecture, binary, conv_enable=True):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        scope, hidden_conv, n_hidden_recog_1, n_hidden_recog_2, n_hidden_gener_1,  n_hidden_gener_2, n_input, n_z = self._unpack_network_arguments(**network_architecture)
        with tf.variable_scope(scope):
            if hidden_conv and conv_enable:
                z_2d = tf.reshape(z, (-1, 1, 1, n_z))
                layer_1 = deconv_2d(    z_2d,
                                        filter_set=[3, n_z, n_hidden_gener_1],
                                        stride=None,
                                        padding='VALID'
                                        )   # no nonlinear transformation
            else:
                weights_hidden_1 = tf.Variable(xavier_init(n_z, n_hidden_recog_1))
                biases_hidden_1 = tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32))
                layer_1 = self.transfer_fct(tf.add(tf.matmul(z, weights_hidden_1),
                                               biases_hidden_1))

            if hidden_conv and conv_enable:
                layer_1_25 = deconv_2d(     layer_1,
                                            filter_set=[5, n_hidden_gener_1, n_hidden_gener_1/2],
                                            stride=1,
                                            padding='VALID'
                                        )   # no nonlinear transformation
                layer_1_50 = deconv_2d(     layer_1_25,
                                            filter_set=[5, n_hidden_gener_1/2, n_hidden_gener_2],
                                            stride=2,
                                            padding='SAME'
                                            )   # no nonlinear transformation
                layer_2_2d = deconv_2d(     layer_1_50,
                                            filter_set=[5, n_hidden_gener_2, 1],
                                            stride=2,
                                            padding='SAME'
                                            )   # no nonlinear transformation
                layer_2 = tf.reshape(layer_2_2d, (-1, n_input))
            else:
                weights_hidden_2 = tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2))
                biases_hidden_2 = tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32))
                layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights_hidden_2),
                                               biases_hidden_2))

            if binary:
                if hidden_conv and conv_enable:
                    weights_genr_mean = tf.Variable(xavier_init(n_input, n_input))
                    biases_genr_mean = tf.Variable(tf.zeros([n_input]))
                    x_reconstr_mean = \
                        tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights_genr_mean),
                                             biases_genr_mean))
                else:
                    weights_genr_mean = tf.Variable(xavier_init(n_hidden_recog_2, n_input))
                    biases_genr_mean = tf.Variable(tf.zeros([n_input]))
                    x_reconstr_mean = \
                        tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights_genr_mean),
                                             biases_genr_mean))
            else:
                weights_genr_mean = tf.Variable(xavier_init(n_hidden_recog_2, n_input))
                biases_genr_mean = tf.Variable(tf.zeros([n_input]))
                x_reconstr_mean = \
                    tf.add(tf.matmul(layer_2, weights_genr_mean),
                                         biases_genr_mean)
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        # <hyin/Apr-1st-2016> 1e-10 seems not enough to regularize for more epoches
        self.vae_costs = []
        self.vae_reconstr_losses = []
        self.vae_latent_losses = []
        for binary, x, x_reconstr_mean, z_mean, z_log_sigma_sq, weight in zip(self.binary, self.x, self.x_reconstr_means, self.z_means, self.z_log_sigma_sqs, self.weights):
            if binary:
                reconstr_loss = \
                    -tf.reduce_sum(x * tf.log(1e-3 + x_reconstr_mean)
                                   + (1-x) * tf.log(1e-3 + 1 - x_reconstr_mean),
                                   1)
            else:
                # <hyin/Apr-4th-2016> square cost for gaussian output, need average?
                reconstr_loss = \
                    tf.reduce_sum(tf.nn.l2_loss(x - x_reconstr_mean))
            # 2.) The latent loss, which is defined as the Kullback Leibler divergence
            ##    between the distribution in latent space induced by the encoder on
            #     the data and some prior. This acts as a kind of regularizer.
            #     This can be interpreted as the number of "nats" required
            #     for transmitting the the latent space distribution given
            #     the prior.
            latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
                                               - tf.square(z_mean)
                                               - tf.exp(z_log_sigma_sq), 1)
            self.vae_reconstr_losses.append(reconstr_loss)
            self.vae_latent_losses.append(latent_loss)
            self.vae_costs.append(tf.reduce_mean(reconstr_loss + latent_loss) * weight)

        #<hyin/Apr-8th-2016> now we have a third term to associate latent representations
        #associativity regularization reconstron the latent representations
        self.assoc_costs = []
        #combine each pair of sensory latent representations
        for i, j in list(combinations(range(len(self.network_architectures)), 2)):
            self.assoc_costs.append(
                # <hyin/Apr-8th-2016> it is wrong to simply expand the univariate case to multi-variate one
                # tf.reduce_sum(
                # tf.div(tf.square(self.z_means[i] -self.z_means[j]) + tf.exp(self.z_log_sigma_sqs[i]) - tf.exp(self.z_log_sigma_sqs[j]),
                #         2 * tf.exp(self.z_log_sigma_sqs[i]) + 1e-3)
                # + 0.5 * (self.z_log_sigma_sqs[i] - self.z_log_sigma_sqs[j])
                # )
                # see http://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians for the derivation
                tf.reduce_sum(
                0.5 * (tf.reduce_sum(self.z_log_sigma_sqs[j], 1) - tf.reduce_sum(self.z_log_sigma_sqs[i], 1) - self.n_z
                + tf.reduce_sum(tf.exp(self.z_log_sigma_sqs[i] - self.z_log_sigma_sqs[j]), 1)
                + tf.reduce_sum(tf.mul(tf.square(self.z_means[j] - self.z_means[i]), tf.exp(-self.z_log_sigma_sqs[j])), 1))
                )
                #<hyin/May-19th-2016> try a symmetry distance
                + tf.reduce_sum(
                0.5 * (tf.reduce_sum(self.z_log_sigma_sqs[i], 1) - tf.reduce_sum(self.z_log_sigma_sqs[j], 1) - self.n_z
                + tf.reduce_sum(tf.exp(self.z_log_sigma_sqs[j] - self.z_log_sigma_sqs[i]), 1)
                + tf.reduce_sum(tf.mul(tf.square(self.z_means[i] - self.z_means[j]), tf.exp(-self.z_log_sigma_sqs[i])), 1))
                )
                )

        if self.assoc_costs:
            self.cost = tf.add_n(self.vae_costs) + self.assoc_lambda * tf.add_n(self.assoc_costs)
        else:
            self.cost = tf.add_n(self.vae_costs)
        # Use ADAM optimizerz_mean
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        return

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={x_sens:X_sens for x_sens, X_sens in zip(self.x, X)})

        return cost

    def evaluate_cost(self, X):
        cost = self.sess.run(self.cost,
                                  feed_dict={x_sens:X_sens for x_sens, X_sens in zip(self.x, X)})
        return cost

    def transform(self, X, sens_idx=None):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        # sens_idx is either None or an integer, indicating the desired data modality
        if sens_idx is None:
            latent_reps = [self.sess.run(z_mean, feed_dict={x_sens: X_sens}) for z_mean, x_sens, X_sens in zip(self.z_means, self.x, X)]
        else:
            assert sens_idx < len(self.z_means)
            latent_reps = self.sess.run(self.z_mean[sens_idx], feed_dict={self.x[sens_idx]: X})
        return latent_reps

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            #<hyin/Apr-8th-2016> the size of random variables have to be same as a batch...
            z_mu = np.random.normal(size=(self.batch_size, self.n_z))
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distributionn_hidden_recog_1
        x_reconstr_means = [self.sess.run(x_reconstr_mean,
                             feed_dict={z: z_mu}) for x_reconstr_mean, z in zip(self.x_reconstr_means, self.z_array)]
        return x_reconstr_means

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        x_reconstr_means = [self.sess.run(x_reconstr_mean,
                            feed_dict={x_sens: X_sens}) for x_reconstr_mean, x_sens, X_sens in zip(self.x_reconstr_means, self.x, X)]
        return x_reconstr_means

    def save_model(self, fname=None):
        if fname is None:
            ts = time.time()
            ckpt_fname = 'vae_assoc_' + datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S') + '_batchsize_{}.ckpt'.format(self.batch_size)
        else:
            ckpt_fname = fname
        print 'Saving model to {}...'.format(ckpt_fname)
        save_path = self.saver.save(self.sess, ckpt_fname)
        return

    def restore_model(self, folder=None, fname=None):
        if folder is None:
            model_folder = 'output'
        else:
            model_folder = folder

        if os.path.isdir(model_folder) and os.path.exists(model_folder):
            if fname is None:
                #search all checkpoint files in the folder
                files = [f for f in os.listdir(model_folder) if f.endswith('.ckpt')]
                if not files:
                    print 'No valid model file.'
                    return
                else:
                    model_file = files[-1]
            else:
                model_file = fname
            if os.path.exists(os.path.join(model_folder, model_file)):
                #using saver to load it
                print 'Loading {}...'.format(os.path.join(model_folder, model_file))
                self.saver.restore(self.sess, os.path.join(model_folder, model_file))
            else:
                print 'Invalid or non-exist model file.'

        else:
            print 'Invalid or non-exist model folder.'
        return

'''
Convolution hidden layer for 2d grayscale image
'''
import prettytensor as pt
from deconv import deconv2d

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# helper functions for convolution and deconvolution operation for 2D mono-channel images
def conv_2d(x_2d, filter_set=[5, 1, 32], stride=2, padding='VALID', transfer_fct=None):
    #hidden layer with a convolution operation
    W_conv = weight_variable([filter_set[0], filter_set[0], filter_set[1], filter_set[2]])
    if transfer_fct is not None:
        h_conv = transfer_fct(tf.nn.conv2d(x_2d, W_conv, strides=(1, stride, stride, 1), padding=padding))
    else:
        h_conv = tf.nn.conv2d(x_2d, W_conv, strides=(1, stride, stride, 1), padding=padding)

    #no pooling for now
    return h_conv

def deconv_2d(x_2d, filter_set=[5, 32, 1], stride=None, padding='VALID', transfer_fct=tf.nn.sigmoid):

    return (pt.wrap(x_2d).
            # reshape([-1, 1, 1, filter_set[1]]).
            deconv2d(kernel=filter_set[0], depth=filter_set[2], stride=stride, edges=padding, activation_fn=transfer_fct)
            ).tensor

def train(data_sets, network_architectures, binary=True, weights=1.0, assoc_lambda=1e-5, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5, early_stop=False):
    vae_assoc = AssocVariationalAutoEncoder(network_architectures,
                                 binary,
                                 transfer_fct=tf.nn.relu,
                                 weights=weights,
                                 assoc_lambda=assoc_lambda,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)

    n_samples = data_sets.train._data.shape[0]

    sens_indices = np.concatenate([[0], np.cumsum([na["n_input"] for na in network_architectures])])
    avg_cost_hist = []

    valid_cost = None

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)

        if early_stop:
            #if it is the time to check early stop condition
            if epoch % early_stop == 0:
                #calculate current validation cost
                curr_valid_cost = 0
                n_valid_batches = int(data_sets.validation._data.shape[0] / batch_size)
                for i in range(n_valid_batches):
                    batch_xs, _ = data_sets.validation.next_batch(batch_size)
                    #construct batch data to desired format: segement different sensory modalities
                    batch_xs_seg = [batch_xs[:, sens_indices[i]:sens_indices[i+1]] for i in range(len(network_architectures))]
                    curr_valid_cost += vae_assoc.evaluate_cost(batch_xs_seg) / n_valid_batches
                print "Validation cost=", "{:.9f}".format(curr_valid_cost)
                if valid_cost is not None:
                    if curr_valid_cost > valid_cost:
                        #stop the iteration
                        print 'Validation error increases. Early stop at epoch {} to prevent overfitting...'.format(epoch+1)
                        break
                valid_cost = curr_valid_cost

        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = data_sets.train.next_batch(batch_size)
            #construct batch data to desired format: segement different sensory modalities
            batch_xs_seg = [batch_xs[:, sens_indices[i]:sens_indices[i+1]] for i in range(len(network_architectures))]

            # print batch_xs_seg[0][-1, :]
            # print vae_assoc.network_weights[0]["weights_recog"]['h1'].eval()
            # print vae_assoc.network_weights[0]["weights_recog"]['h2'].eval()

            # vae_costs = [vae_assoc.sess.run(cost,
            #                           feed_dict={x_sens:X_sens}) for cost, x_sens, X_sens in zip(vae_assoc.vae_costs, vae_assoc.x, batch_xs_seg)]
            # print vae_costs

            # reconstr_costs = [vae_assoc.sess.run(reconstr_cost,
            #                           feed_dict={x_sens:X_sens}) for reconstr_cost, x_sens, X_sens in zip(vae_assoc.vae_reconstr_losses, vae_assoc.x, batch_xs_seg)]
            # latent_costs = [vae_assoc.sess.run(latent_cost,
            #                           feed_dict={x_sens:X_sens}) for latent_cost, x_sens, X_sens in zip(vae_assoc.vae_latent_losses, vae_assoc.x, batch_xs_seg)]
            #
            # print reconstr_costs
            # print latent_costs
            # z_means = [vae_assoc.sess.run(z_mean,
            #                           feed_dict={x_sens:X_sens}) for z_mean, x_sens, X_sens in zip(vae_assoc.z_means, vae_assoc.x, batch_xs_seg)]
            # print z_means
            #
            # z_log_sigma_sqs = [vae_assoc.sess.run(z_log_sigma_sq,
            #                           feed_dict={x_sens:X_sens}) for z_log_sigma_sq, x_sens, X_sens in zip(vae_assoc.z_log_sigma_sqs, vae_assoc.x, batch_xs_seg)]
            # print z_log_sigma_sqs

            # x_reconstr_means = [vae_assoc.sess.run(x_reconstr_mean,
            #                           feed_dict={x_sens:X_sens}) for x_reconstr_mean, x_sens, X_sens in zip(vae_assoc.x_reconstr_means, vae_assoc.x, batch_xs_seg)]
            # print x_reconstr_means
            # raw_input()

            # Fit training using batch data
            cost = vae_assoc.partial_fit(batch_xs_seg)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size
            avg_cost_hist.append(avg_cost)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost)
    return vae_assoc, avg_cost_hist
