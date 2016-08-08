#!/usr/bin/env python
from __future__ import print_function
import sys
import cv2

class CameraImageReader:
    def __init__(self, camera_idx=1):
        self.camera_idx = camera_idx
        self.cap = cv2.VideoCapture(camera_idx)
        if not self.cap.isOpened():
            print('The camera is busy...')
            self.cap.open()
        #the camera needs to warm up to take high quality pictures...
        self.warm_up()
        return

    def __del__(self):
        cv2.VideoCapture(self.camera_idx).release()

    def warm_up(self):
        ramp_frames = 30
        for i in range(ramp_frames):
            temp = self.cap.read()
        return

    def print_cap_info(self):
        test = self.cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
        ratio = self.cap.get(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO)
        frame_rate = self.cap.get(cv2.cv.CV_CAP_PROP_FPS)
        width = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        brightness = self.cap.get(cv2.cv.CV_CAP_PROP_BRIGHTNESS)
        contrast = self.cap.get(cv2.cv.CV_CAP_PROP_CONTRAST)
        saturation = self.cap.get(cv2.cv.CV_CAP_PROP_SATURATION)
        hue = self.cap.get(cv2.cv.CV_CAP_PROP_HUE)
        gain = self.cap.get(cv2.cv.CV_CAP_PROP_GAIN)
        exposure = self.cap.get(cv2.cv.CV_CAP_PROP_EXPOSURE)
        print("Test: ", test)
        print("Ratio: ", ratio)
        print("Frame Rate: ", frame_rate)
        print("Height: ", height)
        print("Width: ", width)
        print("Brightness: ", brightness)
        print("Contrast: ", contrast)
        print("Saturation: ", saturation)
        print("Hue: ", hue)
        print("Gain: ", gain)
        print("Exposure: ", exposure)
        return

    def capture_image(self):
        ret, img = self.cap.read()
        return img

import numpy as np
import utils

class ImageLetterProcessor:
    def __init__(self, img=None):
        self.img = img
        self.img_processed = self.img
        return

    def binarize_img(self):
        if self.img_processed is None:
            print('No available image for processing.')
        else:
            ret,thres = cv2.threshold(self.img_processed,96,255,cv2.THRESH_BINARY)
            # thres = cv2.adaptiveThreshold(self.img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            self.img_processed = 255-thres

        return self.img_processed

    def localize_img(self):
        if self.img_processed is None:
            print('No available image for processing.')
        else:
            bound_rect = utils.segment_char_contour_bounding_box(self.img_processed)
            center = [bound_rect[0] + bound_rect[2]/2., bound_rect[1] + bound_rect[3]/2.]
            #crop the interested part
            leng = max([int(bound_rect[2]), int(bound_rect[3])])
            print(leng)
            border = int(0.6*leng)
            pt1 = int(center[1] -bound_rect[3] // 2)
            pt2 = int(center[0] -bound_rect[2] // 2)
            cv_img_bckgrnd = np.zeros((border+leng, border+leng))
            # print cv_img_bckgrnd.shape
            # print bound_rect
            # print center
            # print border
            # print (pt1+border//2),(pt1+bound_rect[3]+border//2), (pt2+border//2),(pt2+bound_rect[2]+border//2)
            # print cv_img_bckgrnd[(border//2):(bound_rect[3]+border//2), (border//2):(bound_rect[2]+border//2)].shape

            cv_img_bckgrnd[ (border//2+(leng-bound_rect[3])//2):(bound_rect[3]+border//2+(leng-bound_rect[3])//2),
                            (border//2+(leng-bound_rect[2])//2):(bound_rect[2]+border//2+(leng-bound_rect[2])//2)] = self.img_processed[pt1:(pt1+bound_rect[3]), pt2:(pt2+bound_rect[2])]
            self.img_processed = cv_img_bckgrnd
        return self.img_processed

import os
import time

def main(record=True):
    if record:
        img_reader = CameraImageReader(camera_idx=0)
        #print information
        # img_reader.print_cap_info()
        #wait until enter to capture an image
        raw_input('ENTER to capture an image...')

        img = img_reader.capture_image()
        cv2.imshow("capture", img)
        cv2.waitKey()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(os.path.join('img_reader', timestr+'.png'),img)
    else:
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        fdir = os.path.join(curr_dir, 'img_reader')
        fname = '20160808-131404.png'
        #load the colored image in grayscale
        img = cv2.imread(os.path.join(fdir, fname), cv2.IMREAD_GRAYSCALE)
        cv2.imshow("file", img)
        cv2.waitKey()

        img_processor = ImageLetterProcessor(img=img)
        img_processed = img_processor.binarize_img()
        img_processed = img_processor.localize_img()
        cv2.imshow("processed", img_processed)
        cv2.waitKey()

    return

if __name__ == '__main__':
    main(False)
