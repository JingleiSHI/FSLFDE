#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 24/05/2018 14:07
# @Author  : Jinglei SHI
import tensorflow as tf
from fn2.flow_warp import flow_warp
from scipy import ndimage
import numpy as np
class Flow_warper():

    def _conv(self,image): # image shape: height,width,1
        h = image.shape[0].value
        w = image.shape[1].value
        kernel = np.ones(shape=[3,3])/9.
        filtered_image = tf.py_func(lambda x:ndimage.convolve(x,kernel,mode='reflect', cval=0.0),inp=[tf.squeeze(image)],Tout=tf.float32)
        filtered_image.set_shape([h,w])
        return filtered_image  # shape: height,width

    def conv_warping(self,flow,group_image,ref_image):
        """
        :param flow: shape [16,height,width,2]
        :param group_image:  shape [16,height,width,3]
        :param ref_image:  shape [height,width,3]
        :return:
        """
        number_group_image,_,_,_ = flow.shape.as_list()
        extended_image = tf.tile(tf.expand_dims(ref_image,0),[number_group_image,1,1,1])
        warped_image = flow_warp(group_image,flow)
        warping_error = tf.reduce_sum(tf.square(warped_image-extended_image),-1) # shape: [16,height,width]
        conv_warping_error = tf.map_fn(lambda x: self._conv(x),elems=warping_error,dtype=tf.float32)  # shape: [16,height,width]
        min_warping_error = tf.reduce_min(conv_warping_error,0) # shape: [height,width]
        average_warping_error = tf.reduce_mean(conv_warping_error,0) # shape: [height,width]

        return min_warping_error,average_warping_error
    def cal_warping(self,flow_batch,stacked_displacement,stacked_image,ref_image):
        """
        :param flow_batch: shape [batch_size,height,width,1]
        :param stacked_displacement:  shape [batch_size,16,2]
        :param stacked_image: shape [batch_size,16,height,width,3]
        :param ref_image: shape [batch_size,height,width,3]
        :return:
        """
        _, group_image_number,height,width,_ = stacked_image.shape.as_list()
        double_flow = tf.tile(flow_batch,[1,1,1,2]) # shape [batch_size,height,width,2]
        extended_flow = tf.tile(tf.expand_dims(double_flow,1),[1,group_image_number,1,1,1]) # shape: [batch_size,16,height,width,2]
        extended_displacement = tf.tile(tf.expand_dims(tf.expand_dims(stacked_displacement,2),2),[1,1,height,width,1]) # shape: [batch_size,16,height,width,2]
        move_flow = extended_flow*extended_displacement # shape: [batch_size,16,height,width,2]
        move_flow = move_flow[...,::-1]
        min_warping_error,average_warping_error = tf.map_fn(lambda x: self.conv_warping(x[0],x[1],x[2]),elems=(move_flow,stacked_image,ref_image),dtype=(tf.float32,tf.float32))
        return min_warping_error,average_warping_error
    def get_disp_error(self, flow, displacement, warping_displacement, warping_views, ref_image):
        """
        :param flow: [batch_size,n1+n2,height,width,2]
        :param displacement: [batch_size,n1+n2,2], the first dimension is row and the second dimension is column
        :param warping_displacement: [batch_size,n3,2], the first dimension is row and the second dimension is column
        :param warping_views: [batch_size,n3,height,width,3]
        :param ref_image: [batch_size,height,width,3]
        :return:
        """
        batch_size,candid_number,height,width,channel = flow.shape.as_list()
        flow = tf.expand_dims(flow[...,0],-1)
        disp = flow/(tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_sum(displacement,-1,True),3),3),[1,1,height,width,1]))
        transposed_disp = tf.transpose(disp,[1,0,2,3,4])

        min_warping_error,average_warping_error = tf.map_fn(lambda x: self.cal_warping(x,warping_displacement,warping_views,ref_image),elems=transposed_disp,dtype=(tf.float32,tf.float32))
        final_disp = tf.transpose(transposed_disp[...,0],[1,2,3,0])
        final_min_warping_error = tf.transpose(min_warping_error,[1,2,3,0])
        final_average_warping_error = tf.transpose(average_warping_error,[1,2,3,0])
        return final_disp,final_min_warping_error,final_average_warping_error

