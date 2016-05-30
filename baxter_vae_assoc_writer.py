import itertools
import cPickle as cp

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Image
import ImageOps

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty

import baxter_writer as bw

import dataset
import vae_assoc
import pygcem

import utils


class BaxterVAEAssocWriter(bw.BaxterWriter):

    def __init__(self):
        bw.BaxterWriter.__init__(self)

        self.vae_assoc_model = None
        self.initialize_tf_environment()

        self.initialize_dataset()
        return

    def initialize_tf_environment(self):
        self.batch_size = 100
        self.n_z = 10
        self.assoc_lambda = 15

        self.img_network_architecture = \
            dict(scope='image',
                 hidden_conv=False,
                 n_hidden_recog_1=500, # 1st layer encoder neurons
                 n_hidden_recog_2=500, # 2nd layer encoder neurons
                 n_hidden_gener_1=500, # 1st layer decoder neurons
                 n_hidden_gener_2=500, # 2nd layer decoder neurons
                 n_input=784, # MNIST data input (img shape: 28*28)
                 n_z=self.n_z)  # dimensionality of latent space

        self.jnt_network_architecture = \
            dict(scope='joint',
                 hidden_conv=False,
                 n_hidden_recog_1=200, # 1st layer encoder nmodel_batchsize64_nz10_lambda8_weight30eurons
                 n_hidden_recog_2=200, # 2nd layer encoder neurons
                 n_hidden_gener_1=200, # 1st layer decoder neurons
                 n_hidden_gener_2=200, # 2nd lAttempting to use uninitialized valueayer decoder neurons
                 n_input=147, # 21 bases for each function approximator
                 n_z=self.n_z)  # dimensionality of latent space
        # self.img_network_architecture = \
        #     dict(scope='image',
        #          hidden_conv=True,
        #          n_hidden_recog_1=32, # 1st layer encoder neurons - depth for convolution layer
        #          n_hidden_recog_2=128, # 2nd layer encoder neurons - depth for convolution layer
        #          n_hidden_gener_1=128, # 1st layer decoder neurons - depth for convolution layer
        #          n_hidden_gener_2=32, # 2nd layer decoder neurons - depth for convolution layer
        #          n_input=28*28, # MNIST data input (img shape: 28*28)
        #          n_z=self.n_z)  # dimensionality of latent space
        #
        # self.jnt_network_architecture = \
        #     dict(scope='joint',
        #          hidden_conv=False,
        #          n_hidden_recog_1=200, # 1st layer encoder neurons
        #          n_hidden_recog_2=200, # 2nd layer encoder neurons
        #          n_hidden_gener_1=200, # 1st layer decoder neurons
        #          n_hidden_gener_2=200, # 2nd layer decoder neurons
        #          n_input=147, # 21 bases for each function approximator
        #          n_z=self.n_z)  # dimensionality of latent space

        self.initialize_vae_assoc()
        return

    def initialize_vae_assoc(self):
        #close the session of the existing model
        # if self.vae_assoc_model is not None:
        #     self.vae_assoc_model.sess.close()
        self.vae_assoc_model = vae_assoc.AssocVariationalAutoEncoder([self.img_network_architecture, self.jnt_network_architecture],
                                     [True, False],
                                     transfer_fct=tf.nn.relu,
                                     assoc_lambda=self.assoc_lambda,
                                     learning_rate=0.0001,
                                     batch_size=self.batch_size)
        return

    def initialize_dataset(self):
        img_data = utils.extract_images(fname='bin/img_data_extend.pkl', only_digits=False)
        #we need mean and standard deviation to restore the function approximator
        fa_data, self.fa_mean, self.fa_std = utils.extract_jnt_fa_parms(fname='bin/jnt_ik_fa_data_extend.pkl', only_digits=False)

        fa_data_normed = (fa_data - self.fa_mean) / self.fa_std

        # fa_data_sets = dataset.construct_datasets(fa_data_normed)

        #put them together
        aug_data = np.concatenate((img_data, fa_data_normed), axis=1)

        self.data_sets = dataset.construct_datasets(aug_data)
        return

    def train_model(self):
        # if self.vae_assoc_model is not None:
        #     self.vae_assoc_model.sess.close()
        tf.reset_default_graph()
        self.vae_assoc_model, self.cost_hist = vae_assoc.train(self.data_sets, [self.img_network_architecture, self.jnt_network_architecture], binary=[True, False], assoc_lambda = self.assoc_lambda, learning_rate=0.0001,
                    batch_size=self.batch_size, training_epochs=5000, display_step=5)
        return

    def save_model(self):
        if self.vae_assoc_model is not None:
            self.vae_assoc_model.save_model('output/model_batchsize{}_nz{}_lambda{}.ckpt'.format(self.batch_size, self.n_z, self.assoc_lambda))
        return

    def load_model(self, folder=None, fname=None):
        #prepare network and load from a file
        tf.reset_default_graph()
        self.initialize_vae_assoc()
        self.vae_assoc_model.restore_model(folder, fname)
        return

    def cost_robot_motion_fa_img_latent(self, latent_rep, auxargs):
        z_mu = np.zeros((self.batch_size, len(latent_rep)))
        z_mu[0, :] = latent_rep
        x_reconstr_means = self.vae_assoc_model.generate(z_mu=z_mu)
        fa_parms = (x_reconstr_means[1] * self.fa_std + self.fa_mean)[0]
        cost = self.cost_robot_motion_fa_img(fa_parms, auxargs)
        return cost

    def cost_robot_motion_fa_img(self, fa_parms, auxargs):
        jnt_traj = self.derive_jnt_traj_from_fa_parms(np.reshape(fa_parms, (self.robot_dynamics._num_jnts, -1)))
        img_data, spatial_traj = self.derive_img_from_robot_motion(jnt_traj)
        tar_img = auxargs['tar_img']
        #cross entropy loss on the difference of the image
        # img_loss = np.sum(tar_img * np.log(1e-3 + img_data) + (1-tar_img) * np.log(1e-3 + 1 - img_data))
        img_loss = np.linalg.norm(img_data - tar_img)
        height_loss = np.linalg.norm(spatial_traj[:, 2] - auxargs['height'])
        # print img_loss, height_loss
        cost = img_loss + 10 * height_loss

        return cost

    def derive_robot_motion_from_from_img_posterior_rl_latent(self, tar_img, init_latent_rep, n_rollouts=10, n_itrs=10):
        #do posterior RL but explore within the latent representation space...
        z_mu = np.zeros((self.batch_size, len(init_latent_rep)))
        z_mu[0, :] = init_latent_rep
        x_reconstr_means = self.vae_assoc_model.generate(z_mu=z_mu)
        init_fa_parms = (x_reconstr_means[1] * self.fa_std + self.fa_mean)[0]
        jnt_traj = self.derive_jnt_traj_from_fa_parms(np.reshape(init_fa_parms, (self.robot_dynamics._num_jnts, -1)))
        spatial_traj = self.derive_cartesian_trajectory(jnt_traj)
        spatial_traj = np.array(spatial_traj)
        auxargs = {'tar_img':tar_img, 'height': np.mean(spatial_traj[:, 2])}

        #posterior RL procedure to refine the motion
        gcem = pygcem.GaussianCEM(x0=init_latent_rep, eliteness=10, covar_full=False, covar_learning_rate=1, covar_scale=None, covar_bounds=[0.1])
        gcem.covar *= 0.01
        curr_mean = gcem.mean
        print curr_mean

        for itr in range(n_itrs):
            #generate samples to take rollouts
            rollout_latent_reps = gcem.sample(n_samples=n_rollouts)
            costs = [self.cost_robot_motion_fa_img_latent(latent_rep, auxargs) for latent_rep in rollout_latent_reps]
            curr_avg_cost = np.mean(costs)
            print 'Avg cost: ', curr_avg_cost

            gcem.fit(rollout_latent_reps, costs)
            diff = curr_mean - gcem.mean
            if np.linalg.norm(diff) < 1e-6:
                break
            curr_mean = gcem.mean

        z_mu[0, :] = curr_mean
        x_reconstr_means = self.vae_assoc_model.generate(z_mu=z_mu)
        res_fa_parms = (x_reconstr_means[1] * self.fa_std + self.fa_mean)[0]
        return res_fa_parms

    def derive_robot_motion_from_from_img_posterior_rl(self, tar_img, init_fa_parms, n_rollouts=100, n_itrs=5):
        jnt_traj = self.derive_jnt_traj_from_fa_parms(np.reshape(init_fa_parms, (self.robot_dynamics._num_jnts, -1)))
        spatial_traj = self.derive_cartesian_trajectory(jnt_traj)
        spatial_traj = np.array(spatial_traj)
        auxargs = {'tar_img':tar_img, 'height': np.mean(spatial_traj[:, 2])}
        #apply the linear constraint to fix the initial point
        for func_approx, jnt_init in zip(self.func_approxs, jnt_traj[0]):
            func_approx.set_linear_equ_constraints([0], [jnt_init])

        #posterior RL procedure to refine the motion
        gcem = pygcem.GaussianCEM(x0=init_fa_parms, eliteness=10, covar_full=False, covar_learning_rate=1, covar_scale=None, covar_bounds=[0.1])
        gcem.covar *= 0.001
        curr_mean = gcem.mean

        for itr in range(n_itrs):
            #generate samples to take rollouts
            mean_fa_parms = np.reshape(gcem.mean, (self.robot_dynamics._num_jnts, -1))
            covars_fa_parms = np.reshape(np.diag(gcem.covar), (self.robot_dynamics._num_jnts, -1))
            rollout_fa_parms = [func_approx.gaussian_sampling(theta=mean_fa_parm, noise=np.diag(covar_fa_parm), n_samples=n_rollouts)
                for func_approx, mean_fa_parm, covar_fa_parm in zip(self.func_approxs, mean_fa_parms, covars_fa_parms)]
            #concatenate the parms for each dof
            rollout_fa_parms = np.concatenate(rollout_fa_parms, axis=1)
            costs = [self.cost_robot_motion_fa_img(fa_parm, auxargs) for fa_parm in rollout_fa_parms]
            curr_avg_cost = np.mean(costs)
            print 'Avg cost: ', curr_avg_cost

            gcem.fit(rollout_fa_parms, costs)
            diff = curr_mean - gcem.mean
            if np.linalg.norm(diff) < 1e-6:
                break
            curr_mean = gcem.mean

        return curr_mean

    def derive_img_from_robot_motion(self, jnt_traj):
        # to simulate drawing the letter image from the given joint movement
        spatial_traj = self.derive_cartesian_trajectory(jnt_traj)

        fig = plt.figure(frameon=False, figsize=(4,4), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        spatial_traj = np.array(spatial_traj)
        ax.plot(-spatial_traj[:, 1], spatial_traj[:, 0], linewidth=12.0)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])

        ax.set_aspect('equal')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()

        buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )

        buf.shape = ( w, h, 4 )

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll ( buf, 3, axis = 2 )

        #prepare an image
        img = Image.fromstring( "RGBA", ( w ,h ), buf.tostring() )
        img_gs = img.convert('L')

        thumbnail_size = (28, 28)
        img_gs.thumbnail(thumbnail_size)
        img_gs_inv = ImageOps.invert(img_gs)

        img_gs_inv_thumbnail, bound_rect = utils.get_char_img_thumbnail_helper(np.asarray(img_gs_inv))
        # img_gs_inv_thumbnail = img_gs_inv
        # img.show()

        img_data = np.asarray(img_gs_inv_thumbnail).flatten().astype(np.float32) * 1./255.

        plt.close(fig)
        return img_data, spatial_traj

    def derive_img_from_robot_motion_fa(self, fa_parms):
        jnt_traj = self.derive_jnt_traj_from_fa_parms(fa_parms)
        img_data, spatial_traj = self.derive_img_from_robot_motion(jnt_traj)

        return img_data, spatial_traj

    def derive_robot_motion_from_from_img(self, img, posterior_rl=True):
        if self.vae_assoc_model is not None:
            if len(img.shape) == 1:
                assert len(img) == 784
                #construct fake data to pad the batch
                input_img_data = np.random.rand(self.batch_size, 784) - 0.5
                input_img_data[0] = img
            else:
                assert img.shape[0] == self.batch_size
                assert img.shape[1] == 784

                input_img_data = img

            #construct input with fake joint parms
            X = [input_img_data, np.random.rand(self.batch_size, 147) - 0.5]

            #use recognition model to infer the latent representation
            z_rep = self.vae_assoc_model.transform(X)
            #now remember to only use the z_rep of img to
            #generate the joint output
            x_reconstr_means = self.vae_assoc_model.generate(z_mu=z_rep[0])

            if len(img.shape) == 1:
                #take the first joint fa param and restore it for the evaluation
                fa_parms = (x_reconstr_means[1] * self.fa_std + self.fa_mean)[0]
                if posterior_rl:
                    #conduct posterior reinforcement learning based on the initial inference from the latent representations
                    # fa_parms = self.derive_robot_motion_from_from_img_posterior_rl(tar_img=img, init_fa_parms=fa_parms)
                    fa_parms = self.derive_robot_motion_from_from_img_posterior_rl_latent(tar_img=img, init_latent_rep=z_rep[0][0])

                jnt_motion = np.array(self.derive_jnt_traj_from_fa_parms(np.reshape(fa_parms, (7, -1))))
                cart_motion = np.array(self.derive_cartesian_trajectory_from_fa_parms(np.reshape(fa_parms, (7, -1))))
            else:
                #take all the samples if the input is a batch
                fa_parms = (x_reconstr_means[1] * self.fa_std + self.fa_mean)
                jnt_motion = [np.array(self.derive_jnt_traj_from_fa_parms(np.reshape(fa, (7, -1)))) for fa in fa_parms]
                cart_motion = [np.array(self.derive_cartesian_trajectory_from_fa_parms(np.reshape(fa, (7, -1)))) for fa in fa_parms]


            return fa_parms, jnt_motion, cart_motion

        return None, None, None

import os
import sys
import drawing_pad as dp
import threading

from PyQt4.QtCore import *
from PyQt4.QtGui import *


def main(use_gui=False):
    #prepare a writer and load a trained model
    tf.reset_default_graph()
    bvaw = BaxterVAEAssocWriter()

    curr_dir = os.path.dirname(os.path.realpath(__file__))

    bvaw.load_model(os.path.join(curr_dir, 'output/work/non_cnn/1000epoches'), 'model_batchsize64_nz10_lambda8_weight30.ckpt')
    print 'Number of variabels:', len(tf.all_variables())

    #prepare ros stuff
    rospy.init_node('baxter_vaeassoc_writer')
    r = rospy.Rate(100)

    jnt_pub = rospy.Publisher('/baxter_openrave_writer/joint_cmd', JointState, queue_size=10)
    cln_pub = rospy.Publisher('/baxter_openrave_writer/clear_cmd', Empty, queue_size=10)

    if not use_gui:
        n_test = 20

        plt.ion()

        test_sample = bvaw.data_sets.test.next_batch(bvaw.batch_size)[0] #the first is feature, the second is the label
        test_img_sample = test_sample[:, :784]
        fa_motion, jnt_motion, cart_motion = bvaw.derive_robot_motion_from_from_img(test_img_sample)

        raw_input('ENTER to start the test...')

        for i in range(n_test):
            #prepare image to show
            fig = plt.figure()
            ax_img = fig.add_subplot(121)
            ax_img.imshow(test_img_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap='gray')
            ax_img.set_title("Test Image Input")
            # plt.colorbar()

            ax_cart = fig.add_subplot(122)
            ax_cart.plot(-cart_motion[i][:, 1], cart_motion[i][:, 0], linewidth=3.5)
            ax_cart.set_title("Associative Motion")
            ax_cart.set_aspect('equal')

            # print 'z coord mean and std: {}, {}'.format(z_coord_mean, z_coord_std)

            plt.draw()

            print 'Sending joint command to a viewer...'
            cln_pub.publish(Empty())
            for k in range(10):
                r.sleep()
            jnt_msg = JointState()
            for cmd in jnt_motion[i]:
                jnt_msg.position = cmd
                jnt_pub.publish(jnt_msg)
                r.sleep()

            raw_input()
    else:
        app = QApplication(sys.argv)
        dpad = dp.DrawingPad()

        #threading function
        def threading_func(img_data, writer, rate, clean_pub, write_pub):
            fa_motion, jnt_motion, cart_motion = bvaw.derive_robot_motion_from_from_img(img_data)
            print 'Sending joint command to a viewer...'
            cln_pub.publish(Empty())
            for k in range(10):
                rate.sleep()
            jnt_msg = JointState()
            for cmd in jnt_motion:
                jnt_msg.position = cmd
                jnt_pub.publish(jnt_msg)
                rate.sleep()
            return

        #prepare a user callback to send the message
        def send_msg(gui):
            t = threading.Thread(target=threading_func, args = (gui.img_data, bvaw, r, cln_pub, jnt_pub))
            t.daemon = True
            t.start()
            return

        dpad.on_send_usr_callback = send_msg

        dpad.show()
        print 'Start a drawing pad...'
        app.exec_()

    return

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    main(use_gui=True)
