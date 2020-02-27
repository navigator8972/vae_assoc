#!/usr/bin/python

# Copyright (c) 2013-2014, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import PyKDL

import rospy

import baxter_interface

from baxter_kdl.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF

class baxter_kinematics(object):
    """
    Baxter Kinematics with PyKDL
    """
    def __init__(self, limb, robot_desc_file_path=None):
        if robot_desc_file_path is None:
            #load from parameter server is the description URDF is not given
            self._baxter = URDF.from_parameter_server(key='robot_description')
        else:
            self._baxter = URDF.from_desc_file(robot_desc_file_path)

        self._kdl_tree = kdl_tree_from_urdf_model(self._baxter)
        self._base_link = self._baxter.get_root()
        self._tip_link = limb + '_gripper'
        self._tip_frame = PyKDL.Frame()
        self._arm_chain = self._kdl_tree.getChain(self._base_link,
                                                  self._tip_link)

        # Baxter Interface Limb Instances
        if robot_desc_file_path is None:
            #the interface is valid when the parameter server is accessible
            #but not guaranteed when running locally...
            self._limb_interface = baxter_interface.Limb(limb)
            self._joint_names = self._limb_interface.joint_names()
        else:
            self._limb_interface = None
            self._joint_names = {
                'left': ['left_s0', 'left_s1', 'left_e0', 'left_e1',
                         'left_w0', 'left_w1', 'left_w2'],
                'right': ['right_s0', 'right_s1', 'right_e0', 'right_e1',
                          'right_w0', 'right_w1', 'right_w2']
                }

        # self._num_jnts = len(self._joint_names)
        self._num_jnts = self._arm_chain.getNrOfJoints()

        # KDL Solvers
        self._fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self._arm_chain)
        self._fk_v_kdl = PyKDL.ChainFkSolverVel_recursive(self._arm_chain)
        self._ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self._arm_chain)
        self._ik_v_kdl_wdls = PyKDL.ChainIkSolverVel_wdls(self._arm_chain)
        #set the weights of orientation part to zeros
        # weightTS = np.diag([1, 1, 1, 0, 0, 0, 0]).tolist()
        # self._ik_v_kdl_wdls.setWeightTS(weightTS)
        self._ik_p_kdl = PyKDL.ChainIkSolverPos_NR(self._arm_chain,
                                                   self._fk_p_kdl,
                                                   self._ik_v_kdl)
        self._ik_p_kdl_pos = PyKDL.ChainIkSolverPos_NR( self._arm_chain,
                                                        self._fk_p_kdl,
                                                        self._ik_v_kdl_wdls)
        self._jac_kdl = PyKDL.ChainJntToJacSolver(self._arm_chain)
        self._dyn_kdl = PyKDL.ChainDynParam(self._arm_chain,
                                            PyKDL.Vector.Zero())

    def print_robot_description(self):
        nf_joints = 0
        for j in self._baxter.joints:
            if j.type != 'fixed':
                nf_joints += 1
        print( "URDF non-fixed joints: %d;" % nf_joints)
        print( "URDF total joints: %d" % len(self._baxter.joints))
        print( "URDF links: %d" % len(self._baxter.links))
        print( "KDL joints: %d" % self._kdl_tree.getNrOfJoints())
        print( "KDL segments: %d" % self._kdl_tree.getNrOfSegments())
        print( "Arm KDL joints: %d" % self._num_jnts)

    def print_kdl_chain(self):
        for idx in xrange(self._arm_chain.getNrOfSegments()):
            print( '* ' + self._arm_chain.getSegment(idx).getName())

    def joints_to_kdl(self, type, vals=None):
        kdl_array = PyKDL.JntArray(self._num_jnts)
        if vals is not None:
            if len(vals) != self._num_jnts:
                print( 'Invalid length of joint values, please check. Use current values')

        if vals is None or len(vals) != self._num_jnts:
            if self._limb_interface is not None:
                #no or invalid values are provided, use current ones
                if type == 'positions':
                    cur_type_values = self._limb_interface.joint_angles()
                elif type == 'velocities':
                    cur_type_values = self._limb_interface.joint_velocities()
                elif type == 'torques':
                    cur_type_values = self._limb_interface.joint_efforts()
                for idx, name in enumerate(self._joint_names):
                    kdl_array[idx] = cur_type_values[name]
                if type == 'velocities':
                    kdl_array = PyKDL.JntArrayVel(kdl_array)
            else:
                #raise an error
                kdl_array = None
                print( 'Need to specify joint values or have the access to the Baxter interface to use the current values.')
        else:
            for idx in range(len(vals)):
                kdl_array[idx] = vals[idx]
            if type == 'velocities':
                kdl_array = PyKDL.JntArrayVel(kdl_array)

        return kdl_array

    def kdl_to_mat(self, data):
        # mat =  np.mat(np.zeros((data.rows(), data.columns())))
        mat = np.zeros((data.rows(), data.columns()))
        for i in range(data.rows()):
            for j in range(data.columns()):
                mat[i,j] = data[i,j]
        return mat

    def forward_position_kinematics(self, angles=None):
        end_frame = PyKDL.Frame()
        self._fk_p_kdl.JntToCart(self.joints_to_kdl('positions', angles),
                                 end_frame)
        pos = end_frame.p
        rot = PyKDL.Rotation(end_frame.M)
        rot = rot.GetQuaternion()
        return np.array([pos[0], pos[1], pos[2],
                         rot[0], rot[1], rot[2], rot[3]])

    def forward_velocity_kinematics(self):
        end_frame = PyKDL.FrameVel()
        self._fk_v_kdl.JntToCart(self.joints_to_kdl('velocities'),
                                 end_frame)
        return end_frame.GetTwist()

    def inverse_kinematics(self, position, orientation=None, seed=None):
        ik = PyKDL.ChainIkSolverVel_pinv(self._arm_chain)
        pos = PyKDL.Vector(position[0], position[1], position[2])
        if orientation is not None:
            rot = PyKDL.Rotation()
            rot = rot.Quaternion(orientation[0], orientation[1],
                                 orientation[2], orientation[3])
        # Populate seed with current angles if not provided
        seed_array = PyKDL.JntArray(self._num_jnts)
        if seed is not None:
            seed_array.resize(len(seed))
            for idx, jnt in enumerate(seed):
                seed_array[idx] = jnt
        else:
            seed_array = self.joints_to_kdl('positions')

        # Make IK Call
        if orientation is not None:
            goal_pose = PyKDL.Frame(rot, pos)
        else:
            goal_pose = PyKDL.Frame(pos)
        result_angles = PyKDL.JntArray(self._num_jnts)

        if orientation is not None:
            if self._ik_p_kdl.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
                #<hyin/Mar-12th-2015> this is not right way to get an ndarray...
                result = [result_angles[idx] for idx in range(result_angles.rows())]
                return np.array(result)
            else:
                print( 'No IK Solution Found')
                return None
        else:
            #position only
            if self._ik_p_kdl_pos.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
                #<hyin/Mar-12th-2015> this is not right way to get an ndarray...
                result = [result_angles[idx] for idx in range(result_angles.rows())]
                return np.array(result)
            else:
                print( 'No IK Solution Found')
                return None

    def jacobian(self, angles=None):
        jacobian = PyKDL.Jacobian(self._num_jnts)
        self._jac_kdl.JntToJac(self.joints_to_kdl('positions', angles), jacobian)
        return self.kdl_to_mat(jacobian)

    def jacobian_transpose(self, angles=None):
        return self.jacobian(angles).T

    def jacobian_pseudo_inverse(self, angles=None):
        return np.linalg.pinv(self.jacobian(angles))


    def inertia(self, angles):
        inertia = PyKDL.JntSpaceInertiaMatrix(self._num_jnts)
        self._dyn_kdl.JntToMass(self.joints_to_kdl('positions', angles), inertia)
        return self.kdl_to_mat(inertia)

    def cart_inertia(self, angles):
        js_inertia = self.inertia(angles)
        jacobian = self.jacobian()
        return np.linalg.inv(jacobian * np.linalg.inv(js_inertia) * jacobian.T)


"""
<hyin/Jun-29th-2015> Add inherited dynamics class
Actually it is not that logically plausible to make dynamics class as the
derived class of kinematics. The reason of doing this here is to keep the
kinematics code untouched (well, it already contains many revisions) and
the dynamcis might need to use kinematics from time to time.
Thus here the dynamics is actually referring to a name of more complete interface rather than
its physical meaning...
"""
class baxter_dynamics(baxter_kinematics):
    def __init__(self, limb, robot_desc_file_path=None):
        baxter_kinematics.__init__(self, limb, robot_desc_file_path)

        #inverse dynamics chain
        self._dyn_kdl = PyKDL.ChainDynParam(self._arm_chain,
                                    PyKDL.Vector(0, 0, -9.81))

        return

    def kdl_to_array(self, data):
        #from KDL JntArray to ndarray
        res = np.zeros(data.rows())
        for i in range(data.rows()):
            res[i] = data[i]
        return res

    """
    supplemented interfaces for colioris and gravity terms
    """
    def coriolis(self, angles, ang_vel):
        """
        Coriolis term as joint torques. Note this is actually not the tensor of Coriolis
        but the product of the tensor and the joint velocity.
        """
        colioris = PyKDL.JntArray(self._num_jnts)
        self._dyn_kdl.JntToCoriolis(
                        self.joints_to_kdl('positions', angles),
                        #note the JntToCoriolis requires all arguments as JntArray but not JntArrayVel for velocity...
                        self.joints_to_kdl('positions', ang_vel),
                        colioris)
        return self.kdl_to_array(colioris)

    def cart_coriolis(self, angles, ang_vel):
        """
        Coriolis term as the Cartesian force and torques
        """
        js_coriolis = self.coriolis(angles, ang_vel)
        jacobian = self.jacobian()
        return np.linalg.inv(jacobian * np.linalg.inv(js_coriolis) * jacobian.T)

    def gravity(self, angles):
        """
        gravity term as joint torques
        """
        gravity = PyKDL.JntArray(self._num_jnts)
        self._dyn_kdl.JntToGravity(
                        self.joints_to_kdl('positions', angles),
                        gravity)
        return self.kdl_to_array(gravity)

    def cart_gravity(self, angles):
        """
        gravity term as Cartesian force and torques
        """
        js_gravity = self.coriolis(angles)
        jacobian = self.jacobian()
        return np.linalg.inv(jacobian * np.linalg.inv(js_gravity) * jacobian.T)
