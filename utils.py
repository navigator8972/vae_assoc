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
            tmp_fname = 'ascii_{0}_{1:03d}.png'.format(ord(dict_key), d_idx)
            tmp_fpath = os.path.join(output_path_char, tmp_fname)
            if not os.path.exists(tmp_fpath) or overwrite:
                print tmp_fpath
                fig, ax = plot_single_stroke_char_or_digit(d)
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(tmp_fpath, bbox_inches=extent, dpi=100)

            if grayscale:
                #load the image to have a grayscale file
                image=Image.open(tmp_fpath).convert("L")
                image.thumbnail(thumbnail_size)
                inverted_image = ImageOps.invert(image)
                # arr=np.asarray(image)
                # plt.figimage(arr,cmap=cm.Greys_r)

                tmp_fname_grayscale = 'ascii_{0}_{1:03d}_grayscale_thumbnail.png'.format(ord(dict_key), d_idx)
                tmp_fpath_grayscale = os.path.join(gs_output_path_char, tmp_fname_grayscale)
                print 'Generating grayscale image {0}'.format(tmp_fname_grayscale)
                inverted_image.save(tmp_fpath_grayscale)
                # plt.close(fig)

                #get the np array data for this image
                img_data[char_folder].append(np.asarray(inverted_image))

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
