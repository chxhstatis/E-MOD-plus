#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
from math import exp as exp

# basic
slide_dir = './input/'
output_dir = './output/'  # not used
patch_dir = './temp/patch/'
heatmap_dir = './temp/composed/'
csv_dir = './temp/csv'  # not used
display_size = (800, 800)  # size of image windows in the GUI
clinical = False
zooming = 20
NUM_CLASSES = 2  # positive/negative (features)
classes = {0: 'Hyperplasia', 1: 'Mild dysplasia', 2: 'Moderate dysplasia', 3: 'Severe dysplasia'}
english_dict = {1: 'irregular epithelial stratification',
                2: 'loss of polarity of basal cells',
                3: 'drop-shaped rete ridges',
                4: 'increased number of mitotic figures',
                5: 'abnormally superficial mitotic figures',
                6: 'premature keratinization in single cells',
                7: 'keratin pearls within rete ridges',
                8: 'loss of epithelial cell cohesion',
                9: 'abnormal variation in nuclear size',
                10: 'abnormal variation in nuclear shape',
                11: 'abnormal variation in cell size',
                12: 'abnormal variation in cell shape',
                13: 'increased N:C ratio',
                14: 'atypical mitotic figures',
                15: 'increased number and size of nucleoli',
                16: 'hyperchromasia'}
size_dict = {224: 'small', 1024: 'large'}
param_dict = {1: [english_dict[1], size_dict[1024], 1],
              2: [english_dict[2], size_dict[1024], 2],
              3: [english_dict[3], size_dict[1024], 3],
              4: [english_dict[4], size_dict[1024], 4],
              5: [english_dict[6], size_dict[224], 6],
              6: [english_dict[8], size_dict[224], 8],
              7: [english_dict[9], size_dict[1024], 9],
              8: [english_dict[10], size_dict[1024], 10],
              9: [english_dict[11], size_dict[224], 11],
              10: [english_dict[12], size_dict[224], 12],
              11: [english_dict[13], size_dict[1024], 13],
              12: [english_dict[16], size_dict[1024], 16]}

# ordinal logistic regression
coefs_files = pd.read_csv('./config/model_coefs.csv')
intercepts_files = pd.read_csv('./config/model_intercepts.csv')

coefs = coefs_files['x']
intercepts = intercepts_files['x']
if not clinical:
    coefs = coefs[0:12]


def predict_glm(array, t=0):
    for i in range(len(coefs)):
        t = t - coefs[i] * array[i]
    t1 = intercepts[0] + t
    t2 = intercepts[1] + t
    t3 = intercepts[2] + t
    p1 = exp(t1) / (exp(t1) + 1)
    p2 = exp(t2) / (exp(t2) + 1) - p1
    p3 = exp(t3) / (exp(t3) + 1) - p1 - p2
    p4 = 1 - p1 - p2 - p3
    return p1, p2, p3, p4


# test = [26, 31, 11, 18, 38, 51, 16, 31, 25, 20, 34, 22]

# print(predict_glm(test))

# CNN
model_dir = 'saved_model/'  # model of 12 features
EMOD = 'saved_model/'  # model of dysplasia prediction
# one prediction model + 12 feature detection model
# now models are in saved_model
