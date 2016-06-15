from PyQt4.QtCore import *
from PyQt4.QtGui import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.pyplot import cm
matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8)

import sys
import os
import time

import numpy as np
import tensorflow as tf

import baxter_vae_assoc_writer as bvaw

class VAEAssocModelViewer(QMainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.writer = bvaw.BaxterVAEAssocWriter()
        print 'VAE Assoc baxter writer initialized.'
        self.create_main_frame()
        self.create_event_handler()
        return

    def create_main_frame(self):
        self.main_frame = QWidget()

        self.main_hbox = QHBoxLayout()
        vbox_ctrl_pnl = QVBoxLayout()
        vbox_ctrl_pnl.setAlignment(Qt.AlignTop)
        hbox_fig = QHBoxLayout()

        vbox_sb_layout = QVBoxLayout()
        #load button
        self.load_btn = QPushButton('Load Model')
        self.save_btn = QPushButton('Save Figures')

        #slider bars to explore in the latent space
        self.slider_bars = [QSlider() for i in range(self.writer.n_z)]
        for sb in self.slider_bars:
            sb.setOrientation(Qt.Horizontal)
            sb.setMaximum(100)
            sb.setValue(50)
            vbox_sb_layout.addWidget(sb)

        vbox_ctrl_pnl.addWidget(self.load_btn)
        vbox_ctrl_pnl.addWidget(self.save_btn)
        vbox_ctrl_pnl.addLayout(vbox_sb_layout)

        self.main_hbox.addLayout(vbox_ctrl_pnl)

        #figure zone
        self.dpi = 100
        self.img_fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.img_canvas = FigureCanvas(self.img_fig)
        self.img_ax = self.img_fig.add_subplot(111)
        self.img_ax.set_aspect('equal')
        self.img_ax.hold(False)

        self.motion_fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.motion_canvas = FigureCanvas(self.motion_fig)
        self.motion_ax = self.motion_fig.add_subplot(111)
        self.motion_ax.set_aspect('equal')
        self.motion_ax.set_xlim([-1.5, 1.5])
        self.motion_ax.set_ylim([-1.5, 1.5])
        self.motion_ax.hold(False)


        hbox_fig.addWidget(self.img_canvas)
        hbox_fig.addWidget(self.motion_canvas)

        self.main_hbox.addLayout(hbox_fig)

        #set main frame
        self.main_frame.setLayout(self.main_hbox)
        self.setCentralWidget(self.main_frame)
        return

    def create_event_handler(self):
        self.load_btn.clicked.connect(self.on_load_model)
        self.save_btn.clicked.connect(self.on_save_figures)

        #link slider_bars to on_plot event
        for sb in self.slider_bars:
            sb.valueChanged.connect(self.on_plot)
        return

    def on_load_model(self):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        default_dir = os.path.join(curr_dir, 'output/work/non_cnn/1000epoches')
        file_full_path = QFileDialog.getOpenFileName(self, 'Open', default_dir, selectedFilter='*.ckpt')
        if file_full_path:
            folder, fname = os.path.split(str(file_full_path))
            self.writer.load_model(folder, fname)
            print 'Model loaded from {0}'.format(str(file_full_path))

        self.on_plot()
        return

    def on_plot(self):
        scale = 5.0
        z_mu = np.zeros((self.writer.batch_size, len(self.slider_bars)))
        z_mu[0, :] = np.array([float((sb.value()-50))/100 for sb in self.slider_bars])
        z_mu *= scale

        x_reconstr_means = self.writer.vae_assoc_model.generate(z_mu=z_mu)
        fa_parms = (x_reconstr_means[1] * self.writer.fa_std + self.writer.fa_mean)[0]
        jnt_traj = self.writer.derive_jnt_traj_from_fa_parms(np.reshape(fa_parms, (self.writer.robot_dynamics._num_jnts, -1)))
        motion_data = np.array(self.writer.derive_cartesian_trajectory(jnt_traj))

        char_traj = motion_data[:, 0:2]
        char_traj = char_traj - np.mean(char_traj, axis=0)
        char_traj = char_traj / self.writer.scale

        img_data = np.reshape(x_reconstr_means[0][0], (28, 28))

        self.img_ax.imshow(img_data, vmin=0, vmax=1)
        self.motion_ax.plot(-char_traj[:, 1], char_traj[:, 0], 'k', linewidth=12.0)
        self.motion_ax.set_aspect('equal')
        self.motion_ax.set_xlim([-1.5, 1.5])
        self.motion_ax.set_ylim([-1.5, 1.5])

        self.img_canvas.draw()
        self.motion_canvas.draw()
        return

    def on_save_figures(self):
        #time stampe for naming
        timestr = time.strftime("%Y%m%d_%H%M%S")

        curr_dir = os.path.dirname(os.path.realpath(__file__))
        default_dir = os.path.join(curr_dir, 'fig/viewer')

        #image
        img_fig_path = os.path.join(default_dir, 'vae_assoc_viewer_{0}_img.svg'.format(timestr))
        extent = self.img_ax.get_window_extent().transformed(self.img_fig.dpi_scale_trans.inverted())
        self.img_fig.savefig(img_fig_path, bbox_inches=extent)

        #motion
        motion_fig_path = os.path.join(default_dir, 'vae_assoc_viewer_{0}_motion.svg'.format(timestr))
        extent = self.motion_ax.get_window_extent().transformed(self.motion_fig.dpi_scale_trans.inverted())
        self.motion_fig.savefig(motion_fig_path, bbox_inches=extent)
        return

def main():
    app = QApplication(sys.argv)
    gui = VAEAssocModelViewer()
    gui.show()
    app.exec_()
    return


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    tf.reset_default_graph()
    main()
