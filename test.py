#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/06/2019 11:01
# @Author  : Jinglei SHI
import numpy as np
import os
import h5py
import cv2
import argparse
import tensorflow as tf
from pipeline import Pipeline

def test(h5_file_path, row, column, min_radius, max_radius, checkpoint,warping_views_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    lf_name = h5_file_path.split('/')[-1]
    lf_name = lf_name.split('.')[0]
    lf = h5py.File(h5_file_path, 'r')
    images_buffer = lf['image'].value/255.
    original_height = images_buffer.shape[0]
    height = np.int(np.ceil(images_buffer.shape[0]/64.)*64)
    original_width = images_buffer.shape[1]
    width = np.int(np.ceil(images_buffer.shape[1]/64.)*64)
    if (height != width):
        length = np.max([height, width])
    else:
        length = height

    scale = 2.
    dimension = 9
    border_length = 1
    images = np.ones(shape=[np.int(length*scale), np.int(length*scale),3,dimension,dimension])
    for i in range(dimension):
        for j in range(dimension):
            extended_image_buffer = np.zeros([length, length, 3])
            extended_image_buffer[:original_height, :original_width, :] = images_buffer[..., i, j]
            images[..., i, j] = cv2.resize(extended_image_buffer, (np.int(length * scale), np.int(length * scale)), interpolation=cv2.INTER_CUBIC)
    print('images shape is: ' + str(np.shape(images)))

    warping_displacement_list = []
    warping_view_positions = warping_views_list
    for elem in warping_view_positions:
        warping_displacement_list.append([elem[0] - row, elem[1] - column])

    flow_cal_row = []
    flow_cal_column = []
    flow_displacement_row = []
    flow_displacement_column = []
    for dis in range(min_radius, max_radius + 1):
        if row - dis >= (1 + border_length):
            flow_cal_column.append([row - dis, column])
            flow_displacement_column.append([-dis, 0])
        if row + dis <= (dimension - border_length):
            flow_cal_column.append([row + dis, column])
            flow_displacement_column.append([dis, 0])
        if column - dis >= (1 + border_length):
            flow_cal_row.append([row, column - dis])
            flow_displacement_row.append([0, -dis])
        if column + dis <= (dimension - border_length):
            flow_cal_row.append([row, column + dis])
            flow_displacement_row.append([0, dis])

    stereo_view_column = np.stack([images[..., elem[0] - 1, elem[1] - 1] for elem in flow_cal_column], 0)
    stereo_view_row = np.stack([images[..., elem[0] - 1, elem[1] - 1] for elem in flow_cal_row], 0)
    stereo_view_column = stereo_view_column[np.newaxis, ...]
    stereo_view_row = stereo_view_row[np.newaxis, ...]

    stereo_displacement_column = np.stack(flow_displacement_column, 0)
    stereo_displacement_row = np.stack(flow_displacement_row, 0)
    stereo_displacement_column = stereo_displacement_column[np.newaxis, ...]
    stereo_displacement_row = stereo_displacement_row[np.newaxis, ...]

    warping_view = np.stack([images[..., elem[0] - 1, elem[1] - 1] for elem in warping_view_positions], 0)
    warping_view = warping_view[np.newaxis, ...]
    warping_displacement = np.stack(warping_displacement_list, 0)
    warping_displacement = warping_displacement[np.newaxis, ...]

    central_image = images[..., row - 1, column - 1]
    central_image = central_image[np.newaxis, ...]

    print('stereo horizon: ' + str(np.shape(stereo_view_row)) + ' stereo vertical: ' + str(np.shape(stereo_view_column)))
    print('stereo horizon displacement: ' + str(np.shape(stereo_displacement_row)) + ' stereo vertical displacement: ' + str(
        np.shape(stereo_displacement_column)))
    print('warping: ' + str(np.shape(warping_view)))
    print('warping displacement: ' + str(np.shape(warping_displacement)))
    print('horizontal_number: ' + str(stereo_displacement_row.shape[1]))
    print('vertical_number: ' + str(stereo_displacement_column.shape[1]))

    # TODO: This is a hack, we should get rid of this
    inputs = {'target_view': tf.constant(central_image, dtype=tf.float32),
              'stereo_horizon_views': tf.constant(stereo_view_row, dtype=tf.float32),
              'stereo_horizon_displacement': tf.constant(stereo_displacement_row, dtype=tf.float32),
              'stereo_vertical_views': tf.constant(stereo_view_column, dtype=tf.float32),
              'stereo_vertical_displacement': tf.constant(stereo_displacement_column, dtype=tf.float32),
              'warping_views': tf.constant(warping_view, dtype=tf.float32),
              'warping_displacement': tf.constant(warping_displacement, dtype=tf.float32),
              'horizontal_number': stereo_displacement_row.shape[1],
              'vertical_number': stereo_displacement_column.shape[1]
              }
    pipeline = Pipeline()
    predictions = pipeline.model(inputs,False,False,False)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        [pred] = sess.run([predictions])
        output_disp = pred['output_disp'][0, :, :, 0]
        final_disp = cv2.resize(output_disp/scale, (length, length), interpolation=cv2.INTER_CUBIC)
        np.save(lf_name + '_' + str(row) + '_' + str(column) + '_test',final_disp[:original_height, :original_width])
        print('Finished ......')
    return


if __name__ == '__main__':
    checkpoint = './models/dense/dense_flex_depthestim.ckpt'
    lf_file_path = './scenes/stilllife/stilllife.h5'
    row =5
    column =5
    min_radius =3
    max_radius =3
    warping_view_positions = [[5, 2], [5, 8], [2, 5], [8, 5]]
    test(h5_file_path=lf_file_path,row=row,column=column,min_radius=min_radius,max_radius=max_radius,checkpoint=checkpoint,warping_views_list=warping_view_positions)