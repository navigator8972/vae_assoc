"""
module of utilities
"""
import cPickle as cp
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

def extract_single_stroke_chars_and_digits(data):
    res_data = defaultdict(list)
    for dict_key in data.keys():
        if (ord(dict_key[0]) >= ord('a') and ord(dict_key[0]) <= ord('z'))  or \
            (ord(dict_key[0]) >= ord('A') and ord(dict_key[0]) <= ord('Z')) or \
            (ord(dict_key[0]) >= ord('0') and ord(dict_key[0]) <=ord('9')):
            for d in data[dict_key]:
                #check if this is single stroke char
                if len(d) == 1:
                    res_data[dict_key].append(d[0])
    return res_data

def plot_single_stroke_char_or_digit(data):
    #data is a single stroke with the last entry as the time scale...
    fig = plt.figure(frameon=False, figsize=(4,4), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    data_len = len(data[:-1])/2
    ax.plot(data[:data_len], -data[data_len:-1], 'k', linewidth=12.0)
    #<hyin/Feb-9th-2016> we need to carefully define the limits of axes
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    ax.set_aspect('equal')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

    return fig, ax

import Image
import ImageOps
import os.path
import time
from collections import defaultdict

def generate_images_for_chars_and_digits(data, overwrite=False, grayscale=True, thumbnail_size=(28, 28)):
    img_data = defaultdict(list)

    func_path = os.path.dirname(os.path.realpath(__file__))
    folder = 'bin/images'
    gs_folder = 'bin/grayscale'

    output_path = os.path.join(func_path, folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    gs_output_path = os.path.join(func_path, gs_folder)
    if not os.path.exists(gs_output_path):
        os.makedirs(gs_output_path)
    for dict_key in data.keys():
        print 'Processing character or digit {0}'.format(dict_key)
        char_folder = 'char_{0}_{1}'.format(ord(dict_key), dict_key)
        output_path_char = os.path.join(output_path, char_folder)
        if not os.path.exists(output_path_char):
            os.makedirs(output_path_char)

        gs_output_path_char = os.path.join(gs_output_path, char_folder)
        if not os.path.exists(gs_output_path_char):
            os.makedirs(gs_output_path_char)

        for d_idx, d in enumerate(data[dict_key]):
            # print 'Generating images for the {0}-th demonstrations...'.format(d_idx)
            tmp_fname = 'ascii_{0}_{1:04d}.png'.format(ord(dict_key), d_idx)
            tmp_fpath = os.path.join(output_path_char, tmp_fname)
            fig = None
            if not os.path.exists(tmp_fpath) or overwrite:
                print tmp_fpath
                fig, ax = plot_single_stroke_char_or_digit(d)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(tmp_fpath, bbox_inches=extent, dpi=100)

            if grayscale:
                #load the image to have a grayscale file
                # image=Image.open(tmp_fpath).convert("L")
                # inverted_image = ImageOps.invert(image)
                # image.thumbnail(thumbnail_size)
                # arr=np.asarray(image)
                # plt.figimage(arr,cmap=cm.Greys_r)

                tmp_fname_grayscale = 'ascii_{0}_{1:04d}_grayscale_thumbnail.png'.format(ord(dict_key), d_idx)
                tmp_fpath_grayscale = os.path.join(gs_output_path_char, tmp_fname_grayscale)
                if not os.path.exists(tmp_fpath_grayscale) or overwrite:
                    thumbnail = get_char_img_thumbnail(tmp_fpath, tmp_fpath_grayscale)
                    print 'Generating grayscale image {0}'.format(tmp_fname_grayscale)
                else:
                    image = Image.open(tmp_fpath_grayscale)
                    thumbnail = np.asarray(image)
                    # image.close()
                # inverted_image.save(tmp_fpath_grayscale)
                # plt.close(fig)

                #get the np array data for this image
                img_data[char_folder].append(np.asarray(thumbnail))
            if fig is not None:
                plt.close(fig)
            # time.sleep(0.5)
    return img_data

#utilities for computing convenience
from scipy import interpolate

def expand_traj_dim_with_derivative(data, dt=0.01):
    augmented_trajs = []
    for traj in data:
        time_len = len(traj)
        t = np.linspace(0, time_len*dt, time_len)
        if time_len > 3:
            if len(traj.shape) == 1:
                """
                mono-dimension trajectory, row as the entire trajectory...
                """
                spl = interpolate.splrep(t, traj)
                traj_der = interpolate.splev(t, spl, der=1)
                tmp_augmented_traj = np.array([traj, traj_der]).T
            else:
                """
                multi-dimensional trajectory, row as the state variable...
                """
                tmp_traj_der = []
                for traj_dof in traj.T:
                    spl_dof = interpolate.splrep(t, traj_dof)
                    traj_dof_der = interpolate.splev(t, spl_dof, der=1)
                    tmp_traj_der.append(traj_dof_der)
                tmp_augmented_traj = np.vstack([traj.T, np.array(tmp_traj_der)]).T

            augmented_trajs.append(tmp_augmented_traj)

    return augmented_trajs

import dataset as ds

def extract_images(data=None, fname=None, only_digits=True, dtype=np.float32):
    data_dict = data
    if fname is not None:
        #try to load from given fname
        data_dict = cp.load(open(fname, 'rb'))

    def extract_image_helper(d):
        #flatten the image and scale them
        return d.flatten().astype(dtype) * 1./255.
    images = []
    if data_dict is not None:
        for char in sorted(data_dict.keys(), key=lambda k:k[-1]):
            if only_digits and ord(char[-1]) > 57:
                continue
            else:
                images += [extract_image_helper(d) for d in data_dict[char]]
    return np.array(images)

def extract_jnt_trajs(data=None, fname=None, only_digits=True, dtype=np.float32):
    data_dict = data
    if fname is not None:
        #try to load from given fname
        data_dict = cp.load(open(fname, 'rb'))

    def extract_jnt_trajs_helper(d):
        #flatten the image and scale them, is it necessary for joint trajectory, say within pi radians?
        return d.flatten().astype(dtype)
    jnt_trajs = []
    if data_dict is not None:
        for char in sorted(data_dict.keys()):
            if only_digits and ord(char[-1]) > 57:
                continue
            else:
                jnt_trajs += [extract_jnt_trajs_helper(d) for d in data_dict[char]]
    return np.array(jnt_trajs)

def extract_jnt_fa_parms(data=None, fname=None, only_digits=True, dtype=np.float32):
    data_dict = data
    if fname is not None:
        #try to load from given fname
        data_dict = cp.load(open(fname, 'rb'))

    fa_parms = []
    if data_dict is not None:
        for char in sorted(data_dict.keys()):
            if only_digits and ord(char[-1]) > 57:
                continue
            else:
                fa_parms += [d for d in data_dict[char]]
    fa_parms = np.array(fa_parms)
    #Gaussian statistics for potential normalization
    fa_mean = np.mean(fa_parms, axis=0)
    fa_std = np.std(fa_parms, axis=0)
    return fa_parms, fa_mean, fa_std

import cv2
'''
utility to threshold the character image
'''
def threshold_char_image(img):
    #do nothing for now
    return img
'''
utility to segment character contour and get the rectangular bounding box
'''
def segment_char_contour_bounding_box(img):
    ctrs, hier = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    #for blank image
    if len(rects) == 0:
        return [0, 0, img.shape[1], img.shape[0]]
    #rect length-4 array, (rect[0], rect[1]) - lower left corner point, (rect[2], rect[3]) - width, height
    corner_pnts = []
    for rect in rects:
        corner_pnts.append([rect[0], rect[1]])
        corner_pnts.append([rect[0]+rect[2], rect[1]+rect[3]])
    corner_pnts = np.array(corner_pnts)
    l_corner_pnt = np.amin(corner_pnts, axis=0)
    u_corner_pnt = np.amax(corner_pnts, axis=0)
    return [l_corner_pnt[0], l_corner_pnt[1], u_corner_pnt[0]-l_corner_pnt[0], u_corner_pnt[1]-l_corner_pnt[1]]
'''
utility to resize
'''
def get_char_img_thumbnail(img_fname, gs_fname):
    #convert this pil image to the cv one
    cv_img = cv2.imread(img_fname)
    cv_img_gs = cv2.cvtColor(np.array(cv_img), cv2.COLOR_BGR2GRAY)
    cv_img_gs_inv = 255 - cv_img_gs
    #first threshold the img
    thres_img = threshold_char_image(cv_img_gs_inv)
    #then figure out the contour bouding box
    bound_rect = segment_char_contour_bounding_box(thres_img)
    center = [bound_rect[0] + bound_rect[2]/2., bound_rect[1] + bound_rect[3]/2.]
    #crop the interested part
    leng = max([int(bound_rect[2]), int(bound_rect[3])])
    border = int(0.6*leng)
    pt1 = int(center[1] -bound_rect[3] // 2)
    pt2 = int(center[0] -bound_rect[2] // 2)
    cv_img_bckgrnd = np.ones((border+leng, border+leng))
    # print cv_img_bckgrnd.shape
    # print bound_rect
    # print center
    # print border
    # print (pt1+border//2),(pt1+bound_rect[3]+border//2), (pt2+border//2),(pt2+bound_rect[2]+border//2)
    # print cv_img_bckgrnd[(border//2):(bound_rect[3]+border//2), (border//2):(bound_rect[2]+border//2)].shape

    cv_img_bckgrnd[ (border//2+(leng-bound_rect[3])//2):(bound_rect[3]+border//2+(leng-bound_rect[3])//2),
                    (border//2+(leng-bound_rect[2])//2):(bound_rect[2]+border//2+(leng-bound_rect[2])//2)] = cv_img_gs_inv[pt1:(pt1+bound_rect[3]), pt2:(pt2+bound_rect[2])]
    # roi = cv_img_gs_inv[pt1:(pt1+border*2+leng), pt2:(pt2+border*2+leng)]
    # Resize the image
    roi = cv2.resize(cv_img_bckgrnd, (28, 28), interpolation=cv2.INTER_AREA)
    # roi = cv2.dilate(roi, (3, 3))
    #write this image
    cv2.imwrite(gs_fname, roi)
    return roi

import pytrajkin_rxzero as pytk_rz

def get_vel_profile(stroke):
    """
    get the velocity profile for a stroke
    input is an array of position coordinates
    """
    vel = np.diff(stroke, axis=0)
    vel_prf = np.sum(vel**2, axis=-1)**(1./2)
    return vel_prf

def get_continuous_ang(stroke):
    """
    get continous angle profile
    see Continous-Angle-Time-Curve
    """
    vel_vec = np.diff(stroke, axis=0)
    ang = np.arctan2(vel_vec[:, 1], vel_vec[:, 0])
    #compute the change of ang
    ang_diff = np.diff(ang)
    #x and y:
    #x = cos(ang_diff)
    #y = sin(ang_diff)
    x = np.cos(ang_diff)
    y = np.sin(ang_diff)
    thres = 1e-14
    #update delta diff
    for idx in range(len(ang_diff)):
        if x[idx] > thres:
            ang_diff[idx] = np.arctan2(y[idx], x[idx])
        elif x[idx] < -thres and y[idx] > 0:
            ang_diff[idx] = np.arctan2(y[idx], x[idx]) + np.pi
        elif x[idx] < -thres and y[idx] < 0:
            ang_diff[idx] = np.arctan2(y[idx], x[idx]) - np.pi
        elif np.abs(x[idx]) < thres and y[idx] > 0:
            ang_diff[idx] = np.pi/2
        elif np.abs(x[idx]) < thres and y[idx] < 0:
            ang_diff[idx] = -np.pi/2
    cont_ang_prf = ang[0] + np.cumsum(ang_diff)
    return np.concatenate([[ang[0]], cont_ang_prf])

def extend_data_with_lognormal_sampling(data_dict, sample_per_char=100, shift_mean=True):
    #<hyin/May-11th-2016> function to diversify the ujichar data
    #the motivation is that, the single stroke letters are not even thus the trained model
    #tend to not perform well for less observed samples. The idea is to generate locally perturbed
    #letters based on human handwriting, with the developed sampling scheme based upon lognormal model
    #note it is desired to balance the number of samples...
    res_data = defaultdict(list)
    #for shifting the data to center it
    for char in sorted(data_dict.keys()):
        n_samples = int(sample_per_char/(len(data_dict[char]) + 1))
        if n_samples == 0:
            #just get the record and continue
            res_data[char] = data_dict[char]
        else:
            #lets first estimate the lognormal parameters for the letter and then perturb them...
            for traj in data_dict[char]:
                res_data[char] += [traj]
                res_data[char] += extend_data_with_lognormal_sampling_helper(traj, n_samples, shift_mean)
    return res_data

def extend_data_with_lognormal_sampling_helper(char_traj, n_samples, shift_mean):
    #the input char_traj is flattened with the last entry as the time, get the 2D form
    data_len = (len(char_traj) - 1)/2
    t_idx = np.linspace(0, 1.0, data_len)
    #is it necessary to also have noise on this?
    x0 = char_traj[0]
    y0 = char_traj[data_len]
    pos_traj = np.array([char_traj[:data_len], char_traj[data_len:-1]]).T
    #estimate the lognormal parms
    lognorm_parms = np.array(pytk_rz.rxzero_train(pos_traj))
    if np.any(np.isinf(lognorm_parms)):
        print 'Unable to extract lognormal parameters. Only use the original trajectory.'
        return []
    n_comps = len(lognorm_parms)
    #generate noise for each components, considering amplitude (+-20%), start angle(+-20 deg) and straightness(+-10% difference)
    ang_difference = lognorm_parms[:, 5] - lognorm_parms[:, 4]
    noises = np.random.randn(n_samples, n_comps, 3) / 3      #white noise to ensure 99.7% samples are within the specified range...
    parm_noises = np.array([ np.array([noise[:, 0]*.2*lognorm_parms[:, 0], np.zeros(n_comps), np.zeros(n_comps), np.zeros(n_comps),
        noise[:, 1]*np.pi/9, noise[:, 1]*np.pi/9 + noise[:, 2]*.1*ang_difference]).T for noise in noises])
    perturbed_parms = np.array([lognorm_parms + parm_noise for parm_noise in parm_noises])
    #apply the noise, remember to flatten and put back the phase scale...
    res_char_trajs = [np.concatenate([pytk_rz.rxzero_traj_eval(perturbed_parm, t_idx, x0, y0)[0].T.flatten(), [char_traj[-1]]]) for perturbed_parm in perturbed_parms]
    if shift_mean:
        mean_coords =  [np.mean(np.reshape(traj[:-1], (2, -1)).T, axis=0) for traj in res_char_trajs]
        for d_idx in range(len(res_char_trajs)):
            data_len = (len(res_char_trajs[d_idx]) - 1)/2
            res_char_trajs[d_idx][0:data_len] -= mean_coords[d_idx][0]
            res_char_trajs[d_idx][data_len:-1] -= mean_coords[d_idx][1]
    return res_char_trajs
