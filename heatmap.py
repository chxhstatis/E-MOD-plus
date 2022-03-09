#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from config import patch_dir, heatmap_dir, EMOD, display_size
from models import efficientnet_4
import tensorflow as tf
import numpy as np
import os
from PIL import Image

IMG_SIZE = 224


def get_size(image_dir):
    coord_list = []
    max_w = 0
    max_h = 0

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            img = os.path.join(root, file)
            file_name = os.path.split(img)[-1]
            coordinate = (int(file_name.split('_')[0]), int(file_name.split('_')[1].split('.')[0]))
            coord_list.append(coordinate)
            # print(coordinate)

    for coord in coord_list:
        if max_w <= coord[0]:
            max_w = coord[0]
        else:
            pass
    for coord in coord_list:
        if max_h <= coord[1]:
            max_h = coord[1]
        else:
            pass
    return max_w, max_h


def get_coordDict(image_dir):
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            file_full = os.path.join(root, file)
            image_list.append(file_full)

    for image in image_list:
        file_name = os.path.split(image)[-1]
        coordinate = (int(file_name.split('_')[0]), int(file_name.split('_')[1].split('.')[0]))
        image_numDict[coordinate] = image

    # print(image_numDict)


# 224 px
def compose(composed_size, the_label, the_softmax):
    # print(composed_size)
    to_image = Image.new('RGB', (composed_size[0] * IMG_SIZE, composed_size[1] * IMG_SIZE), color=(255, 255, 225))
    for x in range(composed_size[0]+1):
        for y in range(composed_size[1]+1):
            if (x, y) in image_numDict:
                # print('{} is {}'.format((x, y), image_numDict[(x, y)]))
                image = Image.open(image_numDict[(x, y)])
                if os.path.split(image_numDict[(x, y)])[-1] in the_label:
                    softmax = the_softmax[(x, y)]
                    offset = adjust_color(softmax)
                    image = add_transparency(img=image,
                                             factor=0.5,
                                             color=(int(255*offset), 0, 0, 0))
                to_image.paste(image, ((x - 1) * IMG_SIZE, (y - 1) * IMG_SIZE))

            else:
                # print('{} not exit'.format((x,y)))
                image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white')
                to_image.paste(image, ((x - 1) * IMG_SIZE, (y - 1) * IMG_SIZE))
    # to_image.show()
    to_image = to_image.resize(display_size)
    to_image.save(os.path.join(heatmap_dir, 'overall.jpeg'), 'jpeg')


# component of color = w1*s1+w2*s2+w3*s3+w4*s4
def adjust_color(array):
    weight = [0.1, 0.3, 0.6, 1]
    product = sum([x*y for x, y in zip(weight, array)])
    return product


def add_transparency(img, factor, color):
    img = img.convert('RGBA')
    layer = Image.new('RGBA', img.size, color)
    img = Image.blend(img, layer, factor)

    return img


def heatmap():
    composed_size = get_size(os.path.join(patch_dir, '224'))
    get_coordDict(os.path.join(patch_dir, '224'))
    compose(composed_size, lab_list, softmax)
    # write_csv()


def get_model():
    return efficientnet_4.efficient_net_b0()


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
        image = tf.image.resize(image_tensor, [224, 224])

    return image


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model = get_model()
    model.load_weights(filepath=os.path.join(EMOD, 'model'))
    # for loop starts here
    for i in range(1):
        softmax = {}
        lab_list = []
        image_list = []
        image_numDict = {}
        for root, dirs, files in os.walk(os.path.join(patch_dir, '224')):
            for f in files:
                full_f = os.path.join(root, f)
                full_f1 = os.getcwd() + full_f
                if os.path.splitext(full_f1)[-1] == '.jpeg':
                    image_raw = tf.io.read_file(filename=full_f)
                    image_tensor = load_and_preprocess_image(image_raw)
                    image_tensor = tf.expand_dims(image_tensor, axis=0)

                    pred = model(image_tensor, training=False)
                    # print(np.array(pred))
                    idx = tf.math.argmax(pred, axis=-1).numpy()[0]
                    # print('idx = '+str(idx))

                    # probability = pred.numpy()[0][idx]
                    coord = (int(f.split('_')[0]), int(f.split('_')[1].split('.')[0]))
                    softmax[coord] = pred.numpy()[0]
                    lab_list.append(f)

                    '''
                    if not class_id[idx] in dict:
                        dict[class_id[idx]] = 0
                    else:
                        dict[class_id[idx]] = dict[class_id[idx]] + 1
                    '''

                else:
                    pass

        heatmap()
