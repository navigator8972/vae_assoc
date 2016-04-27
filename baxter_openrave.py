
import os

import sys
import copy

import cPickle as cp
import numpy as np
import scipy
import scipy.optimize as sciopt

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
import openravepy as orpy

import baxter_pykdl

import utils

class BaxterWritingOpenRave():
    pen_tip_offset = 0.14
    neutral_pos_ = [0.00230, -0.77274, 0.95529, 1.53091, -0.91924, 0.55914, -0.09664]
    writing_start_jnt_ = [0.00230, -0.77274, 0.95529, 1.53091, -0.91924, 0.55914, -0.09664]
    #
    # char_layout_ = {
    #                 'a':[0, 1, 1],          #the first field indicates which cell does the highest point locates
    #                 'b':[-1, 2, 1],         #the second field denotes the cells occupied along vertical direction
    #                 'c':[0, 1, 1],          #the third field denotes the cells occupied along horizontal direction (redundant?)
    #                 'd':[-1, 2, 1],
    #                 'e':[0, 1, 1],
    #                 'f':[-1, 3, 1],
    #                 'g':[0, 2, 1],
    #                 'h':[-1, 2, 1],
    #                 'i':[0, 1, 1],
    #                 'k':[-1, 2, 1],
    #                 'l':[-1, 2, 1],
    #                 'm':[0, 1, 1],
    #                 'n':[0, 1, 1],
    #                 'o':[0, 1, 1],
    #                 'p':[0, 2, 1],
    #                 'q':[0, 2, 1],
    #                 'r':[0, 1, 1],
    #                 's':[0, 1, 1],
    #                 't':[0, 1, 1],
    #                 'u':[0, 1, 1],
    #                 'v':[0, 1, 1],
    #                 'w':[0, 1, 1],
    #                 'x':[0, 1, 1],
    #                 'y':[0, 2, 1],
    #                 'z':[0, 1, 1],
    #                 'A':[-1, 2, 1],
    #                 'B':[-1, 2, 1],
    #                 'C':[-1, 2, 1],
    #                 'D':[-1, 2, 1],
    #                 'E':[-1, 2, 1],
    #                 'F':[-1, 2, 1],
    #                 'G':[-1, 2, 1],
    #                 'H':[-1, 2, 1],
    #                 'I':[-1, 2, 1],
    #                 'J':[-1, 2, 1],
    #                 'K':[-1, 2, 1],
    #                 'L':[-1, 2, 1],
    #                 'M':[-1, 2, 1],
    #                 'N':[-1, 2, 1],
    #                 'O':[-1, 2, 1],
    #                 'P':[-1, 2, 1],
    #                 'Q':[-1, 2, 1],
    #                 'R':[-1, 2, 1],
    #                 'S':[-1, 2, 1],
    #                 'T':[-1, 2, 1],
    #                 'U':[-1, 2, 1],
    #                 'V':[-1, 2, 1],
    #                 'W':[-1, 2, 1],
    #                 'X':[-1, 2, 1],
    #                 'Y':[-1, 2, 1],
    #                 'Z':[-1, 2, 1]
    #                 }

    def __init__(self, manip_idx=1):
        self.robot, self.manip = self.prepare_baxter_robot_manipulator(manip_idx)
        return

    def prepare_baxter_robot_manipulator(self, manip_idx=1):
        #default is the right one: 1
        self.env = orpy.Environment()

        path_prefix = os.path.dirname(os.path.abspath(__file__))
        #urdf
        self.baxter_urdf_path = os.path.join(path_prefix, 'urdf/baxter.urdf')

        #initialize logging handlers
        orpy.misc.InitOpenRAVELogging()

        env_path = os.path.join(path_prefix, 'openrave/baxter_lab.env.xml')
        self.env.Load(env_path)
        self.robot = self.env.GetRobots()[0]

        self.manip = self.robot.GetManipulators()[manip_idx]
        if manip_idx == 0:
            chain_name = 'left_arm'
            kin_name = 'left'
        else:
            chain_name = 'right_arm'
            kin_name = 'right'
        self.robot.SetActiveManipulator(chain_name)
        self.manip_pose = self.manip.GetBase().GetTransform()

        #prepare inverse kinematics model
        self.robot_dynamics = baxter_pykdl.baxter_pykdl.baxter_dynamics(kin_name, self.baxter_urdf_path)
        #print structure
        self.robot_dynamics.print_robot_description()

        dof_limits = self.robot.GetDOFLimits()
        self.dof_upper_limits = dof_limits[0][self.manip.GetArmIndices()]
        self.dof_lower_limits = dof_limits[1][self.manip.GetArmIndices()]

        self.pentip_handle = None
        self.traj_handle = None
        self.traj_data = None

        return self.robot, self.manip


    def set_manipulator_jnt_vals(self, jnt_vals):
        self.robot.SetDOFValues(jnt_vals, self.manip.GetArmIndices())
        return


    def get_pen_tip_from_joint_dof_vals(self, jnt_vals):
        with self.robot:
            self.robot.SetDOFValues(jnt_vals, self.manip.GetArmIndices())
            ef_pose = self.manip.GetEndEffectorTransform()

        #note the offset of pen-tip...
        ef_pose[0:3, 3] += ef_pose[0:3, 0:3].dot(np.array([0, 0, self.pen_tip_offset]))

        return ef_pose

    def openrave_draw_letter_traj(self, letter_traj, color=(0, 0, 1)):
        handles = []
        # print letter_traj
        for strk in letter_traj:
            handles.append(self.env.drawlinestrip(points=strk, linewidth=3.0, colors=np.array([color, ]*len(strk))))

        return handles

    def openrave_draw_inc(self, pnt, color=(0, 0, 1)):
        #draw traj incrementally...
        self.traj_data.append(pnt)
        self.traj_handle = None
        self.traj_handle = self.env.drawlinestrip(points=np.array(self.traj_data), linewidth=3.0, colors=np.array([color, ]*len(self.traj_data)))
        return

    def openrave_clear_traj(self):
        self.traj_data = []
        self.traj_handle = None
        return

    def openrave_draw_pen_barrel(self, handle=None):
        #clear the previous handles
        handle = None
        #get current pose
        with self.robot:
            ef_pose = self.manip.GetEndEffectorTransform()
            start_pos = ef_pose[0:3, 3]
            end_pos = start_pos + ef_pose[0:3, 0:3].dot(np.array([0, 0, self.pen_tip_offset]))
            handle = self.env.drawarrow(p1=start_pos, p2=end_pos, linewidth=0.01, color=[0.2, 0.8, 0.2])

        return handle

    def openrave_sim(self):
        # rospy.init_node('baxter_openrave')
        env = self.env
        env.StopSimulation()
        #add this robot
        # env.Add(baxter_robot)
        cameraRot = orpy.rotationMatrixFromAxisAngle([-0.472792, -0.713688, 0.516833], 2.5656)
        Tstart = np.concatenate([cameraRot, [np.zeros(3)]], axis=0)
        Tstart = np.concatenate([Tstart, [[0.94666], [0.350879], [1.65288], [1]]], axis=1)
        # Tstart=np.array([   [ 0.9983323 ,  0.03651881,  0.04471023, -1.427967],
        #                     [-0.02196539, -0.47593777,  0.87920462,  2.333576],
        #                     [ 0.0533868 , -0.87872044, -0.47434189,  2.153447],
        #                     [ 0.        ,  0.        ,  0.        ,  1.        ]])
        env.SetViewer('qtcoin')
        env.GetViewer().SetCamera(Tstart,
            #focalDistance=0.826495
            )

        #set initial posture
        self.set_manipulator_jnt_vals(self.writing_start_jnt_)

        self.pentip_handle = self.openrave_draw_pen_barrel(self.pentip_handle)
        return

    def close_openrave(self):
        self.env.Destroy()
        orpy.RaveDestroy()
        return

    def joint_msg_handler(self, msg):
        self.set_manipulator_jnt_vals(msg.position)
        ef_pose = self.get_pen_tip_from_joint_dof_vals(msg.position)
        self.openrave_draw_inc(pnt=ef_pose[0:3, 3])
        self.pentip_handle = self.openrave_draw_pen_barrel(self.pentip_handle)
        return

    def clear_msg_handler(self, msg):
        #call clean method
        self.openrave_clear_traj()
        #set initial posture
        self.set_manipulator_jnt_vals(self.writing_start_jnt_)

        self.pentip_handle = self.openrave_draw_pen_barrel(self.pentip_handle)
        return

def main(replay=False):
    #save the opt_res
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    # baxter_mdl_path = os.path.join(path_prefix, '../openrave/baxter.robot.xml')
    # baxter_robot = self.env.ReadRobotXMLFile(baxter_mdl_path)

    #urdf
    # res_pickle_path = os.path.join(path_prefix, './res/opt_res.p')

    #prepare a baxter writer...
    baxter_openrave_writer = BaxterWritingOpenRave()
    #get start point...
    # baxter_openrave_writer.set_manipulator_jnt_vals(baxter_openrave_writer.writing_start_jnt_)
    # start_pose = baxter_openrave_writer.get_pen_tip_from_joint_dof_vals(baxter_openrave_writer.writing_start_jnt_)
    baxter_openrave_writer.openrave_sim()

    #ros node
    rospy.init_node('baxter_writer_openrave')
    r = rospy.Rate(100)

    jnt_sub = rospy.Subscriber('/baxter_openrave_writer/joint_cmd', JointState, baxter_openrave_writer.joint_msg_handler)
    cln_sub = rospy.Subscriber('/baxter_openrave_writer/clear_cmd', Empty, baxter_openrave_writer.clear_msg_handler)

    while not rospy.is_shutdown():
        r.sleep()
    return

if __name__ == '__main__':
    main()
