import itertools
import cPickle as cp
import time

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import baxter_writer as bw

import dataset
import vae_assoc

import utils

np.random.seed(0)
tf.set_random_seed(0)

print 'Loading image data...'
img_data = utils.extract_images(fname='bin/img_data_extend.pkl', only_digits=False)
# img_data = utils.extract_images(fname='bin/img_data.pkl', only_digits=False)
# img_data_sets = dataset.construct_datasets(img_data)
print 'Loading joint motion data...'
fa_data, fa_mean, fa_std = utils.extract_jnt_fa_parms(fname='bin/jnt_ik_fa_data_extend.pkl', only_digits=False)
# fa_data, fa_mean, fa_std = utils.extract_jnt_fa_parms(fname='bin/jnt_fa_data.pkl', only_digits=False)
#normalize data
fa_data_normed = (fa_data - fa_mean) / fa_std

# fa_data_sets = dataset.construct_datasets(fa_data_normed)
print 'Constructing dataset...'
#put them together
aug_data = np.concatenate((img_data, fa_data_normed), axis=1)

data_sets = dataset.construct_datasets(aug_data, validation_ratio=.1, test_ratio=.1)
print 'Start training...'
batch_sizes = [64]
#n_z_array = [3, 5, 10, 20]
n_z_array = [5, 10, 15]
# assoc_lambda_array = [1, 3, 5, 10]
# assoc_lambda_array = [.1, .3, .5]
#assoc_lambda_array = [15, 40]
assoc_lambda_array = [8]
#weights_array = [[2, 1], [5, 1], [10, 1]]
weights_array=[[30, 1], [50, 1]] #work: 30

for batch_size, n_z, assoc_lambda, weights in itertools.product(batch_sizes, n_z_array, assoc_lambda_array, weights_array):

    img_network_architecture = \
        dict(scope='image',
             hidden_conv=False,
             n_hidden_recog_1=500, # 1st layer encoder neurons
             n_hidden_recog_2=500, # 2nd layer encoder neurons
             n_hidden_gener_1=500, # 1st layer decoder neurons
             n_hidden_gener_2=500, # 2nd layer decoder neurons
             n_input=784, # MNIST data input (img shape: 28*28)
             n_z=n_z)  # dimensionality of latent space

    jnt_network_architecture = \
        dict(scope='joint',
             hidden_conv=False,
             n_hidden_recog_1=200, # 1st layer encoder neurons
             n_hidden_recog_2=200, # 2nd layer encoder neurons
             n_hidden_gener_1=200, # 1st layer decoder neurons
             n_hidden_gener_2=200, # 2nd layer decoder neurons
             n_input=147, # 21 bases for each function approximator
             n_z=n_z)  # dimensionality of latent space
    # img_network_architecture = \
    #     dict(scope='image',
    #          hidden_conv=True,
    #          n_hidden_recog_1=16, # 1st layer encoder neurons - depth for convolution layer
    #          n_hidden_recog_2=64, # 2nd layer encoder neurons - depth for convolution layer
    #          n_hidden_gener_1=64, # 1st layer decoder neurons - depth for convolution layer
    #          n_hidden_gener_2=16, # 2nd layer decoder neurons - depth for convolution layer
    #          n_input=28*28, # MNIST data input (img shape: 28*28)
    #          n_z=n_z)  # dimensionality of latent space
    #
    # jnt_network_architecture = \
    #     dict(scope='joint',
    #          hidden_conv=False,
    #          n_hidden_recog_1=200, # 1st layer encoder neurons
    #          n_hidden_recog_2=200, # 2nd layer encoder neurons
    #          n_hidden_gener_1=200, # 1st layer decoder neurons
    #          n_hidden_gener_2=200, # 2nd layer decoder neurons
    #          n_input=147, # 21 bases for each function approximator
    #          n_z=n_z)  # dimensionality of latent space

    #create a new graph to ensure the resource is released after the training
    #change to clear the default graph
    tf.reset_default_graph()
    vae_assoc_model, cost_hist = vae_assoc.train(data_sets, [img_network_architecture, jnt_network_architecture], binary=[True, False], weights=weights, assoc_lambda = assoc_lambda, learning_rate=0.001,
              batch_size=batch_size, training_epochs=1000, display_step=5)

    vae_assoc_model.save_model('output/model_batchsize{}_nz{}_lambda{}_weight{}.ckpt'.format(batch_size, n_z, assoc_lambda, weights[0]))

    # time.sleep(10)
    # #change to clear the default graph
    # tf.reset_default_graph()
    #
    # vae_assoc_model = vae_assoc.AssocVariationalAutoEncoder([img_network_architecture, jnt_network_architecture],
    #                              binary=[True, False],
    #                              transfer_fct=tf.nn.relu,
    #                              assoc_lambda=5,
    #                              learning_rate=0.0001,
    #                              batch_size=batch_size)
    # vae_assoc_model.restore_model()

    x_sample = data_sets.test.next_batch(batch_size)[0]
    #extract image and fa
    x_sample_seg = [x_sample[:, :784], x_sample[:, 784:]]
    x_reconstruct = vae_assoc_model.reconstruct(x_sample_seg)
    x_synthesis = vae_assoc_model.generate()
    z_test = vae_assoc_model.transform(x_sample_seg)

    #restore by scale back
    x_sample_restore = x_sample_seg
    x_sample_restore[1] = x_sample_restore[1] * fa_std + fa_mean
    x_reconstruct_restore = x_reconstruct
    x_reconstruct_restore[1] = x_reconstruct[1] * fa_std + fa_mean
    x_synthesis_restore = x_synthesis
    x_synthesis_restore[1] = x_synthesis[1] * fa_std + fa_mean


    #prepare cartesian samples...
    writer = bw.BaxterWriter()
    # cart_sample = [np.array(writer.derive_cartesian_trajectory(np.reshape(s, (7, -1)).T)) for s in x_sample]
    # cart_reconstruct = [np.array(writer.derive_cartesian_trajectory(np.reshape(s, (7, -1)).T)) for s in x_reconstruct]

    cart_sample = [np.array(writer.derive_cartesian_trajectory_from_fa_parms(np.reshape(s, (7, -1)))) for s in x_sample_restore[1]]
    cart_reconstruct = [np.array(writer.derive_cartesian_trajectory_from_fa_parms(np.reshape(s, (7, -1)))) for s in x_reconstruct_restore[1]]
    cart_synthesis = [np.array(writer.derive_cartesian_trajectory_from_fa_parms(np.reshape(s, (7, -1)))) for s in x_synthesis_restore[1]]

    plt.figure(figsize=(25, 16))
    cart_z_sample = []
    cart_z_reconstr = []
    cart_z_synthesis = []

    for i in range(5):
        #evaluate the Cartesian movement
        # plt.subplot(5, 2, 2*i + 1, projection='3d')
        # plt.plot(xs=cart_sample[i][:, 0], ys=cart_sample[i][:, 1], zs=cart_sample[i][:, 2], linewidth=3.5)
        plt.subplot(5, 6, 6*i + 1)
        plt.plot(-cart_sample[i][:, 1], cart_sample[i][:, 0], linewidth=3.5)
        plt.title("Test joint input")
        plt.axis('equal')
        cart_z_sample = np.concatenate([cart_z_sample, cart_sample[i][:, 2]])
        # plt.subplot(5, 2, 2*i + 2, projection='3d')
        # plt.plot(xs=cart_reconstruct[i][:, 0], ys=cart_reconstruct[i][:, 1], zs=cart_reconstruct[i][:, 2], linewidth=3.5)
        plt.subplot(5, 6, 6*i + 2)
        plt.plot(-cart_reconstruct[i][:, 1], cart_reconstruct[i][:, 0], linewidth=3.5)
        plt.title("Reconstruction joint")
        plt.axis('equal')
        cart_z_reconstr = np.concatenate([cart_z_reconstr, cart_reconstruct[i][:, 2]])

        plt.subplot(5, 6, 6*i + 3)
        plt.imshow(x_sample_restore[0][i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Test image input")
        plt.colorbar()
        plt.subplot(5, 6, 6*i + 4)
        plt.imshow(x_reconstruct_restore[0][i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Reconstruction image")
        plt.colorbar()

        plt.subplot(5, 6, 6*i + 5)
        plt.imshow(x_synthesis_restore[0][i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Synthesis image")
        plt.colorbar()

        plt.subplot(5, 6, 6*i + 6)
        plt.plot(-cart_synthesis[i][:, 1], cart_synthesis[i][:, 0], linewidth=3.5)
        plt.title("Synthesis joint")
        plt.axis('equal')
        cart_z_synthesis = np.concatenate([cart_z_synthesis, cart_synthesis[i][:, 2]])

    plt.savefig('output/samples_batchsize{}_nz{}_lambda{}_weight{}.svg'.format(batch_size, n_z, assoc_lambda, weights[0]))

    print 'Sample Z Coord:', np.mean(cart_z_sample), np.std(cart_z_sample)
    print 'Reconstr Z Coord:', np.mean(cart_z_reconstr), np.std(cart_z_reconstr)
    print 'Synthesis Z Coord:', np.mean(cart_z_synthesis), np.std(cart_z_synthesis)

    print 'Test Latent variables:'
    print 'images:'
    print z_test[0][:5]
    print 'joints:'
    print z_test[1][:5]
    # plt.tight_layout()

    #see the latent representations layout
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28*ny, 28*nx))

    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            #check the first two dimension
            z_mu = np.zeros((batch_size, n_z))
            z_mu[0, 0] = xi
            z_mu[0, 1] = yi
            x_mean = vae_assoc_model.generate(z_mu)
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0][0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper")
    plt.tight_layout()
    plt.savefig('output/samples_2d_batchsize{}_nz{}_lambda{}_weight{}.svg'.format(batch_size, n_z, assoc_lambda, weights[0]))

    #plt.show()
    #save cost hist
    cp.dump(cost_hist, open('output/cost_hist_batchsize{}_nz{}_lambda{}_weight{}.pkl'.format(batch_size, n_z, assoc_lambda, weights[0]), 'wb'))
