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
    ax.plot(data[:data_len], -data[data_len:-1], 'k', linewidth=8.0)
    #<hyin/Feb-9th-2016> we need to carefully define the limits of axes
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    ax.set_aspect('equal')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

    return fig, ax

import Image
import os.path
import time

def generate_images_for_chars_and_digits(data, overwrite=False, grayscale=True, thumbnail_size=(28, 28)):
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
                # arr=np.asarray(image)
                # plt.figimage(arr,cmap=cm.Greys_r)

                tmp_fname_grayscale = 'ascii_{0}_{1:03d}_grayscale_thumbnail.png'.format(ord(dict_key), d_idx)
                tmp_fpath_grayscale = os.path.join(gs_output_path_char, tmp_fname_grayscale)
                print 'Generating grayscale image {0}'.format(tmp_fname_grayscale)
                image.save(tmp_fpath_grayscale)
                # plt.close(fig)
            # time.sleep(0.5)
    return