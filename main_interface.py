#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from __future__ import print_function
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import openslide
import numpy as np
from PIL import Image
import os

import json
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
from optparse import OptionParser
import re
import shutil
from unicodedata import normalize
import numpy as np
import scipy.misc
import subprocess
from glob import glob
from multiprocessing import Process, JoinableQueue
import time
import os
import sys
import dicom
import imageio
from imageio import imsave
from imageio import imread
import time

from PIL import Image, ImageDraw, ImageCms
from skimage import color, io
import tensorflow as tf
import shutil

from config import classes, display_size

global imgName


class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()
        style_window_picture = 'QLabel{background:white;color:rgb(0,0,0);font-size:16px;font-weight:bold}'
        style_window_text = 'QLabel{background:white;color:rgb(0,0,0);font-size:16px;font-weight:bold}'
        style_button = 'font-size:14px'

        self.resize(1080, 1080)
        self.setWindowTitle("E-MOD-plus System")

        self.label = QLabel(self)
        self.label.setText("Select Slide")
        self.label.setFixedSize(400, 400)
        self.label.move(10, 100)
        self.label.setStyleSheet(style_window_picture)

        self.label2 = QLabel(self)
        self.label2.setText("View Heatmap")
        self.label2.setFixedSize(400, 400)
        self.label2.move(430, 100)
        self.label2.setStyleSheet(style_window_picture)

        self.label3 = QLabel(self)
        self.label3.setText("View Feature Results")
        self.label3.setFixedSize(400, 400)
        self.label3.move(10, 530)
        self.label3.setStyleSheet(style_window_text)

        self.label4 = QLabel(self)
        self.label4.setText("View Results")
        self.label4.setFixedSize(400, 400)
        self.label4.move(430, 530)
        self.label4.setStyleSheet(style_window_text)

        btn = QPushButton(self)
        btn.setText("Select Slide")
        btn.move(10, 30)
        btn.clicked.connect(self.openimage)
        btn.setStyleSheet(style_button)
        btn.resize(90,60) # width height


        btn2 = QPushButton(self)
        btn2.setText("Prediction")
        btn2.move(430, 30)
        btn2.clicked.connect(self.predict)
        btn2.setStyleSheet(style_button)
        btn2.resize(90, 60)

    def openimage(self):
        self.label.setText("Loading...")

        imgName, imgType = QFileDialog.getOpenFileName(self, "Select Slide", "", "*.svs;;All Files(*)")  # 获取svs图片

        test = openslide.open_slide(imgName)
        img = np.array(test.read_region((0, 0), 0, test.dimensions))
        img = Image.fromarray(img)
        tifImgName = imgName.split(".")[0] + '.tif'
        # print(tifImgName)
        img = img.resize(display_size)
        img.save(tifImgName)

        jpg = QtGui.QPixmap(tifImgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        self.imageName = imgName
        shutil.copyfile(imgName, os.path.join(os.getcwd(), 'svs/temp.svs'))

    def predict(self):
        self.label2.setText("Predicting...")
        self.label3.setText("Predicting...")
        self.label4.setText("Predicting...")
        with open('cut.py') as cut:
            exec(cut.read(), globals(), globals())
        self.label4.setText('patch generated')
        with open('predict.py') as predict:
            exec(predict.read(), globals(), globals())
        self.label4.setText('softmax extracted')
        with open('heatmap.py') as heatmap:
            exec(heatmap.read(), globals(), globals())
        self.label4.setText('heatmap composed')
        # print(p1, p2, p3, p4)  # global
        pred = classes[np.argmax([p1, p2, p3, p4])]
        style_window_text_update = 'QLabel{background:white;color:rgb(0,0,0);font-size:24px;font-weight:bold}'
        self.label4.setStyleSheet(style_window_text_update)
        self.label4.setText('Hyperplasia: {}%\nMild: {}%\nModerate: {}%\nSevere: {}%\n\n'
                            'Prediction: {}'.format(str(round(100*p1, 0)),
                                                    str(round(100*p2, 0)),
                                                    str(round(100*p3, 0)),
                                                    str(round(100*p4, 0)),
                                                    pred))
        # os.remove(os.path.join(os.getcwd(), 'result.txt'))
        the_heatmap = os.path.join('temp', 'composed', 'overall.jpeg')

        display = QtGui.QPixmap(the_heatmap).scaled(self.label.width(), self.label.height())
        self.label2.setPixmap(display)
        self.label3.setText(' '.join(feature_text))

        clear()  # ???


def empty_folder(path):
    for files in os.listdir(path):
        if os.path.isfile(os.path.join(path, files)):
            os.remove(os.path.join(path, files))


def clear():
    os.remove('temp/composed/overall.jpeg')
    for files in os.listdir('input'):
        if files.endswith('.tif'):
            os.remove(os.path.join('input', files))
    empty_folder('svs')
    empty_folder('temp/patch/224')
    empty_folder('temp/patch/1024')


if __name__ == "__main__":
    if not os.path.exists('svs'):
        os.mkdir('svs')
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())
