#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import openslide
from openslide.deepzoom import DeepZoomGenerator
from config import zooming, slide_dir, patch_dir
import os
import numpy as np

if __name__ == '__main__':
    for root, dirs, files in os.walk(slide_dir):
        svs_files = [f for f in files if f.endswith('svs')]
        file = svs_files[0]
        full_f = os.path.join(root, file)
    # print(file)
    slide = openslide.open_slide(full_f)
    # get slide dimensions, zoom levels, and objective information
    Factors = slide.level_downsamples
    Objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    Available = tuple(Objective / x for x in Factors)
    # find highest magnification greater than or equal to 'Desired'
    Mismatch = tuple(x - zooming for x in Available)
    AbsMismatch = tuple(abs(x) for x in Mismatch)
    if len(AbsMismatch) < 1:
        print(full_f + " - Objective field empty!")

    for size in [224, 1024]:
        dz = DeepZoomGenerator(slide, tile_size=size, overlap=0)
        for level in range(dz.level_count - 1, -1, -1):
            ThisMag = Available[0] / pow(2, dz.level_count - (level + 1))
            if zooming > 0:
                if ThisMag != zooming:
                    continue
            cols, rows = dz.level_tiles[level]
            for col in range(cols):
                for row in range(rows):
                    patch_name = os.path.join(patch_dir, '%d/%d_%d.jpeg' % (size, col, row))
                    patch = dz.get_tile(level=level, address=(col, row))
                    gray = patch.convert('L')
                    binary = gray.point(lambda x: 0 if x < 220 else 1, 'F')
                    avg = np.average(binary)
                    if avg <= 0.5:
                        if not os.path.exists(patch_name):
                            patch.save(patch_name)



