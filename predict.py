#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from PIL import Image
from config import param_dict, patch_dir, model_dir, csv_dir, predict_glm, clinical
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from models import efficientnet

global p1, p2, p3, p4
global feature_text

class_id = {0: 'positive', 1: 'negative'}
model_index = 4
feature_text = []


def get_model():
    if model_index == 4:
        return efficientnet.efficient_net_b0()
    else:
        raise ValueError("The model_index does not exist.")


def load_and_preprocess_image(image_raw, data_augmentation=False):
    # decode
    image_tensor = tf.io.decode_image(contents=image_raw, channels=3, dtype=tf.dtypes.float32)

    if data_augmentation:
        image = tf.image.random_flip_left_right(image=image_tensor)
        image = tf.image.resize_with_crop_or_pad(image=image,
                                                 target_height=int(224 * 1.2),
                                                 target_width=int(224 * 1.2))
        image = tf.image.random_crop(value=image, size=[224, 224, 3])
        image = tf.image.random_brightness(image=image, max_delta=0.5)
    else:
        image = tf.image.resize(image_tensor, [img_size, img_size])

    return image


def write_csv(folder, name, row, content):
    file_path = os.path.join(folder, name)
    with open(file_path, 'a+') as f:
        f.write(row)
        f.write(',')
        f.write(str(content))
        f.write('\n')


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # CNN
    softmax_csv = os.path.join(csv_dir, 'softmax.csv')
    if True:  # if there are softmax files, then skip CNN
        scores = []
        for feature in param_dict.keys():
            saved_model = '{}10e-6-LR-{}-{}-{}/model'.format(model_dir,
                                                             param_dict[feature][2],
                                                             param_dict[feature][0],
                                                             param_dict[feature][1])
            model = get_model()
            model.load_weights(filepath=saved_model)

            softmax = []
            if param_dict[feature][1] == 'small':
                img_dir = os.path.join(patch_dir, '224/')
                img_size = 224
            elif param_dict[feature][1] == 'large':
                img_dir = os.path.join(patch_dir, '1024/')
                img_size = 1024
            else:
                raise ValueError("Invalid size.")
            for root, dirs, files in os.walk(img_dir):
                for f in files:
                    full_f = os.path.join(root, f)
                    if os.path.splitext(full_f)[-1] == '.jpeg':
                        image_raw = tf.io.read_file(filename=full_f)
                        image_tensor = load_and_preprocess_image(image_raw)
                        image_tensor = tf.expand_dims(image_tensor, axis=0)

                        predict = model(image_tensor, training=False)
                        # print(np.array(pred))
                        idx = tf.math.argmax(predict, axis=-1).numpy()[0]
                        # print('idx = '+str(idx))

                        probability = predict.numpy()[0][0]  # positive
                        softmax.append(probability)

                        # print("The predicted category of \'{}\' is: {}".format(f, class_id[idx]))
                    else:
                        pass
            # print('{} - The proportion of {} is {}%'.format(1, class_id[0], 100 * index_0 / p))
            # print('{} - The proportion of {} is {}%'.format(1, class_id[1], 100 * index_1 / p))

            mean = np.mean(softmax)
            scores.append(mean)
            feature_text.append('{}: {}%\n'.format(param_dict[feature][0], str(round(100*mean, 0))))
            print('The softmax of {} is {}'.format(param_dict[feature][0], mean))
    else:
        pass

    # logistic
    scores = [100 * x for x in scores]
    if clinical:
        pass
    else:
        pass
    p1, p2, p3, p4 = predict_glm(array=scores)

