"""
A simulated manipulator based upon Baxter robot
to write given letter trajectory
"""

import os

import sys
import copy

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import baxter_pykdl

import pylqr.pylqr_trajctrl as plqrtc

import utils

class BaxterWriter():
    def __init__(self):
        self.prepare_baxter_robot_manipulator(manip_idx=1)

        #the center of block to write the letter... in the world reference frame
        self.block_center = np.array([0.844, -0.357, 0.257])
        self.scale = 0.09 / (1.5 + 1.5) #boundary of block / dim of trajectories
        #this is for the right arm
        self.seed_pos = [0.00230, -0.77274, 0.95529, 1.53091, -0.91924, 0.55914, -0.09664]
        self.seed_pose = self.robot_dynamics.forward_position_kinematics(self.seed_pos)
        return

    def prepare_baxter_robot_manipulator(self, manip_idx=1):
        path_prefix = os.path.dirname(os.path.abspath(__file__))
        #urdf
        self.baxter_urdf_path = os.path.join(path_prefix, 'urdf/baxter.urdf')

        #default is the right one: 1
        if manip_idx == 0:
            kin_name = 'left'
        else:
            kin_name = 'right'
        # use revised baxter_pykdl to create inverse kinemtics model
        self.robot_dynamics = baxter_pykdl.baxter_pykdl.baxter_dynamics(kin_name, self.baxter_urdf_path)
        #print structure
        self.robot_dynamics.print_robot_description()

        return 

    def generate_spatial_trajectory(self, char_traj):
        """
        Member function to generate spatial trajectory from 2D char trajectory through translation/rotation/scaling
        """
        #char_traj is supposed to be a 2D array
        spatial_traj = np.zeros((len(char_traj), 3))
        spatial_traj[:, 0:2] = char_traj * self.scale
        spatial_traj = spatial_traj + self.block_center
        return spatial_traj

    def derive_ik_trajectory(self, spatial_traj):
        # get orientation from the seed pose
        q_seed = self.seed_pos
        q_array = []
        for idx, pnt in enumerate(spatial_traj):
            #solve IK
            q = self.robot_dynamics.inverse_kinematics(pnt, orientation=self.seed_pose[3:7], seed=q_seed)
            if q is None:
                print 'Failed to solve IK at step {0} for desired position {1}.'.format(idx, pnt)
            else:
                q_seed = q
                q_array.append(q)
        return np.array(q_array)

    def derive_cartesian_trajectory(self, q_array):
        #derive cartesian trajectory from q_array
        spatial_traj = []
        for idx, q in enumerate(q_array):
            cart_pose = self.robot_dynamics.forward_position_kinematics(q)
            spatial_traj.append(cart_pose)
        return spatial_traj

    def derive_ilqr_trajectory(self, spatial_traj):
        dt = .01
        #derive joint trajectory from ilqr
        lqr_traj_ctrl = plqrtc.PyLQR_TrajCtrl(R=.01, dt=dt)

        #cost function
        def traj_cost(x, u, t, aux):
            #x is the given joint position, evaluate forward kinematics
            cart_pose = self.robot_dynamics.forward_position_kinematics(x[0:7])
            track_err = np.linalg.norm((cart_pose[0:3] - spatial_traj[t])*np.array([10., 10., 50]))**2
            #control effort from inverse dynamics
            jnt_mass = self.robot_dynamics.inertia(x[:7])
            # jnt_coriolis = self.robot_dynamics.coriolis(x[:7], x[7:])
            # jnt_gravity = self.robot_dynamics.gravity(x[:7])
            # tau = jnt_mass.dot(u) + jnt_coriolis + jnt_gravity
            # tau = jnt_coriolis + jnt_gravity
            tau = jnt_mass.dot(u)
            #control effort from control input
            # tau = u
            ctrl_effort = np.linalg.norm(tau) ** 2
            return track_err + lqr_traj_ctrl.R_ * ctrl_effort

        lqr_traj_ctrl.build_ilqr_general_solver(cost_func=traj_cost, n_dims=7, T=100)

        #prepare initial guess of trajectory, from the IK solution
        q_ik = self.derive_ik_trajectory(spatial_traj)
        x0 = q_ik[0] #init velocity is zero, this is handled in the ilqr_traj_ctrl

        q_dot_ik = np.diff(q_ik, axis=0) / dt
        q_dot_ik = np.vstack([np.zeros(7), q_dot_ik])   #initial velocity is zero
        q_ddot_ik = np.diff(q_dot_ik, axis=0) / dt
        u_array_init = q_ddot_ik
        #test this init
        #note there will be some difference as the trajectory ilqr force the initial velocity as zero
        #while the finite difference is not the case
        # x_array = lqr_traj_ctrl.ilqr_.forward_propagation(np.concatenate([x0, np.zeros(7)]), u_array_init)
        # print q_ik - np.array(x_array)[:, 0:7]

        syn_traj = lqr_traj_ctrl.synthesize_trajectory(x0, u_array_init, n_itrs=25)
        return syn_traj[:, 0:7]

def build_ik_joint_traj_for_chars(data):
    baxter_writer = BaxterWriter()
    res_data = defaultdict(list)

    for c in data.keys():
        print 'Processing character {0}...'.format(c)
        for d in data[c]:
            tmp_char_traj = np.reshape(d[:-1], (2, -1)).T
            tmp_spatial_traj = baxter_writer.generate_spatial_trajectory(tmp_char_traj)

            tmp_q_array = baxter_writer.derive_ik_trajectory(tmp_spatial_traj)

            #note the data is transposed and flattened as the cartesian trajectories
            res_data[c].append(np.array(tmp_q_array).T.flatten())

    return res_data

def check_joint_ik_data(cart_data, jnt_data, n_chars=5, n_samples=1):
    baxter_writer = BaxterWriter()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(True)
    plt.ion()

    check_chars = [np.random.choice(cart_data.keys()) for i in range(n_chars)]
    for c in check_chars:
        check_indices = [np.random.choice(range(len(cart_data[c]))) for i in range(n_samples)]
        #see how it's going for all the samples
        for idx in check_indices:
            tmp_char_traj = np.reshape(cart_data[c][idx][:-1], (2, -1)).T
            tmp_spatial_traj = baxter_writer.generate_spatial_trajectory(tmp_char_traj)

            ax.plot(tmp_spatial_traj[:, 0], tmp_spatial_traj[:, 1], 'k*', linewidth=3.5)

            #reconstruction from joint trajectories...
            tmp_jnt_traj = np.reshape(jnt_data[c][idx], (7, -1)).T
            tmp_cart_array = baxter_writer.derive_cartesian_trajectory(tmp_jnt_traj)
            recons_char_traj = np.array([cart_pose[0:2] for cart_pose in tmp_cart_array])
            z_array = [cart_pose[3] for cart_pose in tmp_cart_array]
            print 'Z - mean and std:', np.mean(z_array), np.std(z_array)

            #show the reconstructed trajectory
            ax.plot(recons_char_traj[:, 0], recons_char_traj[:, 1], 'r', linewidth=3.5)

            plt.draw()
    return


def main():
    baxter_writer = BaxterWriter()

    #prepare a circular path
    n_pnts = 100
    t = np.linspace(0, 2*np.pi, n_pnts)
    a = 1.35
    b = 1.0
    char_traj = np.array([a*np.cos(t), b*np.sin(t)]).T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(True)
    #derive joint trajectory
    spatial_traj = baxter_writer.generate_spatial_trajectory(char_traj)

    ax.plot(spatial_traj[:, 0], spatial_traj[:, 1], 'k*', linewidth=3.5)
    print 'Solving a series of IK problems...'
    q_array = baxter_writer.derive_ik_trajectory(spatial_traj)
    print 'Finished solving IK problems.'
    #restore to cartesian motion
    print 'Reconstructing Cartesian trajectory...'
    cart_array = baxter_writer.derive_cartesian_trajectory(q_array)
    print 'Finished reconstructing Cartesian trajectory.'
    #extract 2D trajectory
    recons_char_traj = np.array([cart_pose[0:2] for cart_pose in cart_array])

    z_array = [cart_pose[2] for cart_pose in cart_array]
    print 'IK - Z - mean and std:', np.mean(z_array), np.std(z_array)

    #show the reconstructed trajectory
    ax.plot(recons_char_traj[:, 0], recons_char_traj[:, 1], 'r', linewidth=3.5)

    raw_input('ENTER to continue derive iLQR optimal control')

    print 'Solving iLQR optimal trajectories'
    q_array_ilqr = baxter_writer.derive_ilqr_trajectory(spatial_traj)
    print 'Finished solving iLQR optimal control'
    #restore to cartesian motion
    print 'Reconstructing Cartesian trajectory...'
    cart_array_ilqr = baxter_writer.derive_cartesian_trajectory(q_array_ilqr)
    print 'Finished reconstructing Cartesian trajectory.'
    #extract 2D trajectory
    recons_char_traj_ilqr = np.array([cart_pose[0:2] for cart_pose in cart_array_ilqr])
    z_array_ilqr = [cart_pose[2] for cart_pose in cart_array_ilqr]
    print 'iLQR - Z - mean and std:', np.mean(z_array_ilqr), np.std(z_array_ilqr)

    ax.plot(recons_char_traj_ilqr[:, 0], recons_char_traj_ilqr[:, 1], 'g', linewidth=3.5)

    plt.show()
    return

if __name__ == '__main__':
    main()