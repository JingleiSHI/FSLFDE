#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25/05/2018 11:15
# @Author  : Jinglei SHI
import tensorflow as tf
import numpy as np
from skimage import feature
from scipy import ndimage
from fn2.flownet2 import FlowNet2
from warper import Flow_warper
from refinement import Refinement

slim = tf.contrib.slim
class Pipeline():
    def __init__(self):
        self.flow_warper = Flow_warper()
        self.refinementer = Refinement()
        self.flownet = FlowNet2()
    def model(self,inputs,trainable=True,reuse=False,istraining=True):
        with tf.variable_scope('FN2_fuse', istraining, reuse=reuse):
            target_view = inputs['target_view']  # shape: batch,height,width,3
            stereo_horizon_views = inputs['stereo_horizon_views']  # shape: batch,n1,height,width,3
            stereo_horizon_displacement = inputs['stereo_horizon_displacement'] # shape: batch,n1,2
            stereo_vertical_views = inputs['stereo_vertical_views']  # shape: batch,n1,height,width,3
            stereo_vertical_displacement = inputs['stereo_vertical_displacement'] # shape: batch,n2,2
            number1 = inputs['horizontal_number']
            number2 = inputs['vertical_number']

            warping_views = inputs['warping_views'] # shape: batch,number,height,width,3
            warping_displacement = inputs['warping_displacement'] # shape: batch,number,2

            extended_horizon_target_view = tf.tile(tf.expand_dims(target_view, 1), [1, number1, 1, 1, 1])  # shape: batch,number1,height,width,3
            extended_vertical_target_view = tf.tile(tf.expand_dims(target_view, 1), [1, number2, 1, 1, 1])  # shape: batch,number2,height,width,3

            hack_prediction = self.flownet.model({'input_a':target_view,'input_b':target_view},trainable=False,reuse=False)
            def ChannelNorm(tensor):
                sq = tf.square(tensor)
                r_sum = tf.reduce_sum(sq, keep_dims=True, axis=3)
                return r_sum
            def h_pred(x1,x2):
                out_flow = self.flownet.model({'input_a':x1,'input_b':x2},False,True)
                return out_flow['flow']
            def v_pred(x1,x2):
                turned_x1 = tf.map_fn(lambda img: tf.image.rot90(img, 1), x1)
                turned_x2 = tf.map_fn(lambda img: tf.image.rot90(img, 1), x2)
                out_flow = self.flownet.model({'input_a':turned_x1,'input_b':turned_x2},False,True)
                turned_out_flow = tf.map_fn(lambda img: tf.image.rot90(img,-1), out_flow['flow'])
                return turned_out_flow
            def get_grid(x):
                batch_size, height, width, filters = tf.unstack(tf.shape(x))
                Bg, Yg, Xg = tf.meshgrid(tf.range(batch_size), tf.range(height), tf.range(width), indexing='ij')
                return Bg, Yg, Xg
            def percentage_mask(x, percentage):  # error shape: height,width,channel
                threshold = tf.contrib.distributions.percentile(x, percentage)
                mask = 1 - tf.to_float(tf.less(x, threshold * tf.ones_like(x)))
                return mask
            def get_mask(x):
                x = x[..., 0]
                contours = feature.canny(x, sigma=2)
                contours = np.float32(contours)
                mask = ndimage.binary_dilation(contours, structure=np.ones([3, 3])).astype(contours.dtype)
                mask = mask[..., np.newaxis]
                return mask
            h_flow = tf.map_fn(lambda x: h_pred(x[0],x[1]),elems=(tf.transpose(extended_horizon_target_view,[1,0,2,3,4]),tf.transpose(stereo_horizon_views,[1,0,2,3,4])),dtype=tf.float32) # shape: number1,batch,height,width,3
            v_flow = tf.map_fn(lambda x: v_pred(x[0],x[1]),elems=(tf.transpose(extended_vertical_target_view,[1,0,2,3,4]),tf.transpose(stereo_vertical_views,[1,0,2,3,4])),dtype=tf.float32) # shape: number2,batch,height,width,3
            concat_stereo_flow = tf.concat([tf.transpose(h_flow,[1,0,2,3,4]),tf.transpose(v_flow,[1,0,2,3,4])],1) # shape: batch,number1+number2,height,width,3
            concat_stereo_displacement = tf.concat([stereo_horizon_displacement,stereo_vertical_displacement],1) # shape: batch,number1+number2,2

            disp_volume,min_warping_error,average_warping_error = self.flow_warper.get_disp_error(concat_stereo_flow,concat_stereo_displacement,warping_displacement,warping_views,target_view)
            # shape: [batch_size,height,width,n1+n2]

            min_error_index = tf.cast(tf.argmin(min_warping_error,-1),tf.int32)
            average_error_index = tf.cast(tf.argmin(average_warping_error,-1),tf.int32)
            Bg, Xg, Yg = get_grid(disp_volume)
            disp_min_error = tf.gather_nd(disp_volume,tf.stack([Bg,Xg,Yg,min_error_index],-1))
            disp_average_error = tf.gather_nd(disp_volume, tf.stack([Bg, Xg, Yg, average_error_index],-1))
            average_error_map = tf.reduce_min(average_warping_error,-1,keep_dims=True)
            disp_min_error = tf.expand_dims(disp_min_error,-1)
            disp_average_error = tf.expand_dims(disp_average_error,-1)

            fusion_mask = tf.map_fn(lambda x: percentage_mask(x, 85), elems=average_error_map, dtype=tf.float32)
            fusion_disp = disp_min_error * fusion_mask + disp_average_error * (1 - fusion_mask) # batch,height,width,1

            ##################################################################################################################################
            mask_input_disp = tf.map_fn(lambda x: tf.py_func(get_mask, [x], tf.float32), elems=fusion_disp, dtype=tf.float32)
            mask_input_disp.set_shape(fusion_disp.get_shape())
            mask_image = tf.map_fn(lambda x: tf.py_func(get_mask, [x], tf.float32), elems=target_view, dtype=tf.float32)
            mask_image.set_shape(fusion_disp.get_shape())
            artifacts_mask = tf.abs(mask_input_disp - mask_input_disp * mask_image)

            refinement_input = {'disp':fusion_disp,'image':target_view,'mask':artifacts_mask}
            prediction = self.refinementer.net(inputs=refinement_input, trainable=trainable)
            return prediction

    def loss(self,disp,prediction):
        return
