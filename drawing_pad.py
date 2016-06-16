import sys
import os
import time
import cPickle as cp

from PyQt4.QtCore import *
from PyQt4.QtGui import *

import Image
import ImageOps

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8)

import utils

class DrawingPad_Painter(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)

        #prepare data and figure
        self.traj_pnts = []
        self.curr_traj = None
        self.lines = []
        self.curr_line = None

        self.dpi = 100
        self.fig = Figure(figsize=(3.24, 5.0), dpi=self.dpi, facecolor="white")
        self.canvas = FigureCanvas(self.fig)
        self.ax_painter = self.fig.add_subplot(111, aspect='equal')
        self.ax_painter.hold(True)
        self.ax_painter.set_xlim([-2, 2])
        self.ax_painter.set_ylim([-2, 2])
        self.ax_painter.set_aspect('equal')
        self.ax_painter.set_xticks([])
        self.ax_painter.set_yticks([])
        self.ax_painter.axis('off')

        self.hbox_layout = QHBoxLayout()
        self.hbox_layout.addWidget(self.canvas, 5)

        self.line_width = 12.0

        # self.ctrl_pnl_layout = QVBoxLayout()
        # #a button to clear the figure
        # self.clean_btn = QPushButton('Clear')
        # self.ctrl_pnl_layout.addWidget(self.clean_btn)
        #
        # self.hbox_layout.addLayout(self.ctrl_pnl_layout, 1)

        self.setLayout(self.hbox_layout)
        self.drawing = False

        self.create_event_handler()
        return

    def create_event_handler(self):
        self.canvas_button_clicked_cid = self.canvas.mpl_connect('button_press_event', self.on_canvas_mouse_clicked)
        self.canvas_button_released_cid = self.canvas.mpl_connect('button_release_event', self.on_canvas_mouse_released)
        self.canvas_motion_notify_cid = self.canvas.mpl_connect('motion_notify_event', self.on_canvas_mouse_move)

        return

    def on_canvas_mouse_clicked(self, event):
        # print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        # event.button, event.x, event.y, event.xdata, event.ydata)
        self.drawing = True
        # create a new line if we are drawing within the area
        if event.xdata is not None and event.ydata is not None and self.curr_line is None and self.curr_traj is None:
            self.curr_line, = self.ax_painter.plot([event.xdata], [event.ydata], '-k', linewidth=self.line_width)
            self.curr_traj = [np.array([event.xdata, event.ydata])]
            self.canvas.draw()
        return

    def on_canvas_mouse_released(self, event):
        # print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        # event.button, event.x, event.y, event.xdata, event.ydata)
        self.drawing = False
        # store finished line and trajectory
        # print self.curr_traj
        self.lines.append(self.curr_line)
        self.traj_pnts.append(self.curr_traj)
        self.curr_traj = None
        self.curr_line = None
        return

    def on_clean(self, event):
        print 'clean the canvas...'
        #clear everything
        for line in self.ax_painter.lines:
            self.ax_painter.lines.remove(line)
        self.lines = []

        if self.curr_line is not None:
            self.ax_painter.lines.remove(self.curr_line)
        self.curr_line = None
        self.canvas.draw()

        self.traj_pnts = []
        self.curr_traj = None
        self.drawing = False
        return

    def on_canvas_mouse_move(self, event):
        if self.drawing:
            # print 'In movement: x=',event.x ,', y=', event.y,', xdata=',event.xdata,', ydata=', event.ydata
            if event.xdata is not None and event.ydata is not None and self.curr_line is not None and self.curr_traj is not None:
                #append new data and update drawing
                self.curr_traj.append(np.array([event.xdata, event.ydata]))
                tmp_curr_data = np.array(self.curr_traj)
                self.curr_line.set_xdata(tmp_curr_data[:, 0])
                self.curr_line.set_ydata(tmp_curr_data[:, 1])
                self.canvas.draw()
        return

    def plot_trajs_helper(self, trajs):
        tmp_lines = []
        for traj in trajs:
            tmp_line, = self.ax_painter.plot(traj[:, 0], traj[:, 1], '-.g', linewidth=self.line_width)
            tmp_lines.append(tmp_line)
            self.canvas.draw()
        #add these tmp_lines to lines record
        self.lines = self.lines + tmp_lines
        return

    def get_traj_data(self):
        return self.traj_pnts

    def get_image_data(self):
        """
        Get the deposited image
        """
        w,h = self.canvas.get_width_height()
        buf = np.fromstring ( self.canvas.tostring_argb(), dtype=np.uint8 )
        buf.shape = ( w, h, 4 )

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll ( buf, 3, axis = 2 )
        return buf

class DrawingPad(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.resize(540, 360)
        self.move(400, 200)

        #create painter
        self.main_frame = QWidget()
        self.main_hbox = QHBoxLayout()
        self.painter = DrawingPad_Painter()
        self.main_hbox.addWidget(self.painter)
        self.main_frame.setLayout(self.main_hbox)
        self.setCentralWidget(self.main_frame)
        self.setWindowTitle('DrawingPad')

        self.ctrl_pnl_layout = QVBoxLayout()
        #<hyin/Jun-3rd-2016> two more canvas are needed to show the target image as well as the searching ones
        self.fig_target = Figure(figsize=(1.62, 2.5), dpi=self.painter.dpi, facecolor="white")
        self.fig_explore = Figure(figsize=(1.62, 2.5), dpi=self.painter.dpi, facecolor="white")

        self.target_canvas = FigureCanvas(self.fig_target)
        self.explore_canvas = FigureCanvas(self.fig_explore)

        self.ax_target = self.fig_target.add_subplot(111, aspect='equal')
        self.ax_explore = self.fig_explore.add_subplot(111, aspect='equal')

        self.ax_target.set_aspect('equal')
        self.ax_target.set_xticks([])
        self.ax_target.set_yticks([])
        self.ax_target.axis('off')

        self.ax_explore.set_aspect('equal')
        self.ax_explore.set_xticks([])
        self.ax_explore.set_yticks([])
        self.ax_explore.axis('off')

        self.monitor_hbox_layout = QVBoxLayout()
        self.monitor_hbox_layout.addWidget(self.target_canvas)
        self.monitor_hbox_layout.addWidget(self.explore_canvas)

        self.ctrl_pnl_layout.addLayout(self.monitor_hbox_layout, 3)

        #clean button
        #a button to clear the figure
        self.clean_btn = QPushButton('Clear')
        self.ctrl_pnl_layout.addWidget(self.clean_btn)
        self.clean_btn.clicked.connect(self.painter.on_clean)

        #send button
        self.send_btn = QPushButton('Send')
        self.ctrl_pnl_layout.addWidget(self.send_btn)
        self.send_btn.clicked.connect(self.on_send_button_clicked)

        #send corrupted image button
        self.send_incomplete_btn = QPushButton('Send Incomplete')
        self.ctrl_pnl_layout.addWidget(self.send_incomplete_btn)
        self.send_incomplete_btn.clicked.connect(self.on_send_incomplete_button_clicked)

        #save the process data button
        self.save_btn = QPushButton('Save')
        self.ctrl_pnl_layout.addWidget(self.save_btn)
        self.save_btn.clicked.connect(self.on_save)

        self.main_hbox.addLayout(self.ctrl_pnl_layout, 3)

        self.img_data = None
        self.on_send_usr_callback = None

        self.img_incomplete_data = None
        self.fraction_idx = None
        self.on_send_incomplete_usr_callback = None

        #record for potential saving...
        self.record = None
        return

    def on_send_button_clicked(self, event):
        img_data = self.painter.get_image_data()
        #prepare an image
        w, h, d = img_data.shape
        img = Image.fromstring( "RGBA", ( w ,h ), img_data.tostring() )
        img_gs = img.convert('L')
        # thumbnail_size = (28, 28)
        # img_gs.thumbnail(thumbnail_size)
        img_gs_inv = ImageOps.invert(img_gs)
        img_gs_inv_thumbnail, bound_rect = utils.get_char_img_thumbnail_helper(np.asarray(img_gs_inv))
        # img.show()
        self.img_data = np.asarray(img_gs_inv_thumbnail).flatten().astype(np.float32) * 1./255.
        if self.on_send_usr_callback is not None:
            self.on_send_usr_callback(self)
            #update the target canvas
            self.ax_target.imshow(self.img_data.reshape((28, 28)))
            self.target_canvas.draw()
        return

    def on_send_incomplete_button_clicked(self, event):
        img_data = self.painter.get_image_data()
        #prepare an image
        w, h, d = img_data.shape
        img = Image.fromstring( "RGBA", ( w ,h ), img_data.tostring() )
        img_gs = img.convert('L')
        # thumbnail_size = (28, 28)
        # img_gs.thumbnail(thumbnail_size)
        img_gs_inv = ImageOps.invert(img_gs)
        img_gs_inv_thumbnail, bound_rect = utils.get_char_img_thumbnail_helper(np.asarray(img_gs_inv))
        # img.show()
        self.img_data = np.asarray(img_gs_inv_thumbnail).flatten().astype(np.float32) * 1./255.

        #cover a fraction of the image to produce incomplete data
        self.fraction_idx = (np.random.rand(2) > 0.5).astype(int)
        self.img_incomplete_data = self.img_data.reshape((28, 28))
        self.img_incomplete_data[14*self.fraction_idx[0]:14*(self.fraction_idx[0]+1), 14*self.fraction_idx[1]:14*(self.fraction_idx[1]+1)] = 0
        self.img_incomplete_data = self.img_incomplete_data.flatten()

        if self.on_send_incomplete_usr_callback is not None:
            self.on_send_incomplete_usr_callback(self)
            #update the target canvas
            self.ax_target.imshow(self.img_data.reshape((28, 28)))
            self.target_canvas.draw()
        return

    def on_update_explore_canvas(self, img):
        #callback routine to update the exploring canvas
        self.ax_explore.imshow(img.reshape((28, 28)))
        self.explore_canvas.draw()
        return

    def on_save(self):
        if self.record is None:
            print 'The record is not initialized.'
        else:
            #time stampe for naming
            timestr = time.strftime("%Y%m%d_%H%M%S")
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            default_dir = os.path.join(curr_dir, 'res')
            fpath = os.path.join(default_dir, 'res_{0}.pkl'.format(timestr))
            cp.dump(self.record, open(fpath, 'wb'))
            print 'Results saved into {0}'.format(fpath)
        return

    def set_record(self, rec):
        self.record = rec
        return

import sys

def main():
    app = QApplication(sys.argv)
    gui = DrawingPad()
    gui.show()
    app.exec_()
    return


if __name__ == '__main__':
    main()
