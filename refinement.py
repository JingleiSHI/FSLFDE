#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 06/04/2018 11:12
# @Author  : Jinglei SHI
import tensorflow as tf
from fn2.utils import LeakyReLU,pad,antipad
from fn2.downsample import downsample
slim = tf.contrib.slim

class Refinement():
    def net(self, inputs, trainable=True):
            fusion_disp = inputs['disp']
            image = inputs['image']
            ###############################################################################################
            artifacts_mask = inputs['mask']
            ###############################################################################################
            input_to_fusion = tf.concat([image,fusion_disp,artifacts_mask],axis=3)
            input_to_fusion = tf.check_numerics(input_to_fusion,message='input contains NaN')
            with tf.variable_scope('Refinement'):
                with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],trainable=trainable,
                                    weights_initializer=slim.variance_scaling_initializer(),
                                    activation_fn=LeakyReLU,
                                    padding='VALID'):
                    weights_regularizer = slim.l2_regularizer(0.000000004)
                    with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                        with slim.arg_scope([slim.conv2d]):
                            conv1 = slim.conv2d(pad(input_to_fusion, 2), 64, 5, scope='disp_conv1')  # 512
                            conv2 = slim.conv2d(pad(conv1, 2), 128, 5, 2, scope='disp_conv2')  # 256
                            conv2_1 = slim.conv2d(pad(conv2), 128, 3, scope='disp_conv2_1')  # 256
                            conv3 = slim.conv2d(pad(conv2_1), 256, 3, 2, scope='disp_conv3')  # 128
                            conv3_1 = slim.conv2d(pad(conv3), 256, 3, scope='disp_conv3_1')  # 128
                            conv4 = slim.conv2d(pad(conv3_1), 512, 3, 2, scope='disp_conv4')  # 64
                            conv4_1 = slim.conv2d(pad(conv4), 512, 3, scope='disp_conv4_1')  # 64
                            conv5 = slim.conv2d(pad(conv4_1), 1024, 3, 2, scope='disp_conv5')  # 32
                            conv5_1 = slim.conv2d(pad(conv5), 1024, 3, scope='disp_conv5_1')  # 32

                            with slim.arg_scope([slim.conv2d_transpose], biases_initializer=None):
                                res_16 = slim.conv2d(pad(conv5_1), 1, 3, scope='disp_res16', activation_fn=None)  # 32
                                size = [res_16.shape[1],res_16.shape[2]]
                                pr_s1_16 = downsample(fusion_disp,size)
                                pr_s2_16 = pr_s1_16 + res_16

                                up_pred5 = antipad(slim.conv2d_transpose(pr_s2_16, 1, 4, stride=2, scope='disp_upsample_pr_s2_16', activation_fn=None))  # 64
                                upconv4 = antipad(slim.conv2d_transpose(conv5_1, 512, 4, 2, scope='disp_upconv4'))  # 64
                                iconv4_input = tf.concat([upconv4, conv4_1, up_pred5], axis=3)  # 512+512+2
                                iconv4 = slim.conv2d(pad(iconv4_input), 512, 3, scope='disp_iconv4')
                                res_8 = slim.conv2d(pad(iconv4), 1, 3, scope='disp_res8', activation_fn=None)
                                size = [res_8.shape[1],res_8.shape[2]]
                                pr_s1_8 = downsample(fusion_disp,size)
                                pr_s2_8 = pr_s1_8 + res_8

                                up_pred4 = antipad(slim.conv2d_transpose(pr_s2_8, 1, 4, stride=2, scope='disp_upsample_pr_s2_8', activation_fn=None))  # 128
                                upconv3 = antipad((slim.conv2d_transpose(iconv4, 256, 4, 2, scope='disp_upconv3')))  # 128
                                iconv3_input = tf.concat([upconv3, conv3_1, up_pred4], axis=3)  # 256+256+2
                                iconv3 = slim.conv2d(pad(iconv3_input), 256, 3, scope='disp_iconv3')
                                res_4 = slim.conv2d(pad(iconv3), 1, 3, scope='disp_res4', activation_fn=None)
                                size = (res_4.shape[1],res_4.shape[2])
                                pr_s1_4 = downsample(fusion_disp,size)
                                pr_s2_4 = pr_s1_4 + res_4

                                up_pred3 = antipad(slim.conv2d_transpose(pr_s2_4, 1, 4, stride=2, scope='disp_upsample_pr_s2_4', activation_fn=None))  # 256
                                upconv2 = antipad(slim.conv2d_transpose(iconv3, 128, 4, 2, scope='disp_upconv2'))
                                iconv2_input = tf.concat([upconv2, conv2_1, up_pred3], axis=3)  # 128+128+2
                                iconv2 = slim.conv2d(pad(iconv2_input), 128, 3, scope='disp_iconv2')
                                res_2 = slim.conv2d(pad(iconv2), 1, 3, scope='disp_res2', activation_fn=None)
                                size = [res_2.shape[1],res_2.shape[2]]
                                pr_s1_2 = downsample(fusion_disp,size)
                                pr_s2_2 = pr_s1_2 + res_2

                                up_pred2 = antipad(slim.conv2d_transpose(pr_s2_2, 1, 4, stride=2, scope='disp_upsample_pr_s2_2', activation_fn=None))

                                upconv1 = antipad(slim.conv2d_transpose(iconv2, 64, 4, 2, scope='disp_upconv1'))
                                iconv1 = tf.concat([upconv1, conv1, up_pred2], axis=3)  # 64+64+4
                                res_1 = slim.conv2d(pad(iconv1, 2), 1, 5, scope='disp_res1', activation_fn=None)
                                pred_disp = fusion_disp + res_1
            return {
                'output_disp': pred_disp
            }

    def loss(self, disp, predictions):
        return











