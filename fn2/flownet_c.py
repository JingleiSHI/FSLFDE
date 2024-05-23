from .utils import LeakyReLU,pad,antipad
from .correlation import correlation
import tensorflow as tf
slim = tf.contrib.slim

class FlowNetC():
    def model(self, inputs,trainable=True,reuse = False):
        _, height, width, _ = inputs['input_a'].shape.as_list()
        with tf.variable_scope('FlowNetC',reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                trainable=trainable,
                                weights_initializer=slim.variance_scaling_initializer(),
                                activation_fn=LeakyReLU,
                                padding='VALID'):
                with slim.arg_scope([slim.conv2d]):
                    with slim.arg_scope([slim.conv2d], stride=2):
                        conv_a_1 = slim.conv2d(pad(inputs['input_a'], 3), 64, 7, scope='conv1')
                        conv_a_2 = slim.conv2d(pad(conv_a_1, 2), 128, 5, scope='conv2')
                        conv_a_3 = slim.conv2d(pad(conv_a_2, 2), 256, 5, scope='conv3')

                        conv_b_1 = slim.conv2d(pad(inputs['input_b'], 3),64, 7, scope='conv1', reuse=True)
                        conv_b_2 = slim.conv2d(pad(conv_b_1, 2), 128, 5, scope='conv2', reuse=True)
                        conv_b_3 = slim.conv2d(pad(conv_b_2, 2), 256, 5, scope='conv3', reuse=True)

                        cc = correlation(conv_a_3, conv_b_3, 1, 20, 1, 2, 20)
                        cc_relu = LeakyReLU(cc)

                    netA_conv = slim.conv2d(conv_a_3, 32, 1, scope='conv_redir')
                    net = tf.concat([netA_conv, cc_relu], axis=3)

                    conv3_1 = slim.conv2d(pad(net), 256, 3, scope='conv3_1')
                    with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3):
                        conv4 = slim.conv2d(pad(conv3_1), stride=2, scope='conv4')
                        conv4_1 = slim.conv2d(pad(conv4), scope='conv4_1')
                        conv5 = slim.conv2d(pad(conv4_1), stride=2, scope='conv5')
                        conv5_1 = slim.conv2d(pad(conv5), scope='conv5_1')
                    conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='conv6')
                    conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='conv6_1')

                    """ START: Refinement Network """
                    with slim.arg_scope([slim.conv2d_transpose], biases_initializer=None):
                        predict_flow6 = slim.conv2d(pad(conv6_1), 2, 3,scope='predict_flow6',activation_fn=None)
                        deconv5 = antipad(slim.conv2d_transpose(conv6_1, 512, 4,stride=2,scope='deconv5'))
                        upsample_flow6to5 = antipad(slim.conv2d_transpose(predict_flow6, 2, 4,stride=2,scope='upsample_flow6to5',activation_fn=None))
                        concat5 = tf.concat([conv5_1, deconv5, upsample_flow6to5], axis=3)

                        predict_flow5 = slim.conv2d(pad(concat5), 2, 3,scope='predict_flow5',activation_fn=None)
                        deconv4 = antipad(slim.conv2d_transpose(concat5, 256, 4,stride=2,scope='deconv4'))
                        upsample_flow5to4 = antipad(slim.conv2d_transpose(predict_flow5, 2, 4,stride=2,scope='upsample_flow5to4',activation_fn=None))
                        concat4 = tf.concat([conv4_1, deconv4, upsample_flow5to4], axis=3)

                        predict_flow4 = slim.conv2d(pad(concat4), 2, 3,scope='predict_flow4',activation_fn=None)
                        deconv3 = antipad(slim.conv2d_transpose(concat4, 128, 4,stride=2,scope='deconv3'))
                        upsample_flow4to3 = antipad(slim.conv2d_transpose(predict_flow4, 2, 4,stride=2,scope='upsample_flow4to3',activation_fn=None))
                        concat3 = tf.concat([conv3_1, deconv3, upsample_flow4to3], axis=3)

                        predict_flow3 = slim.conv2d(pad(concat3), 2, 3,scope='predict_flow3',activation_fn=None)
                        deconv2 = antipad(slim.conv2d_transpose(concat3, 64, 4,stride=2,scope='deconv2'))
                        upsample_flow3to2 = antipad(slim.conv2d_transpose(predict_flow3, 2, 4,stride=2,scope='upsample_flow3to2',activation_fn=None))
                        concat2 = tf.concat([conv_a_2, deconv2, upsample_flow3to2], axis=3)

                        predict_flow2 = slim.conv2d(pad(concat2), 2, 3,scope='predict_flow2',activation_fn=None)
                    """ END: Refinement Network """

                    flow = predict_flow2 * 20.0
                    # TODO: Look at Accum (train) or Resample (deploy) to see if we need to do something different
                    flow = tf.image.resize_bilinear(flow,tf.stack([height, width]),align_corners=True)

                    return {
                        'predict_flow6': predict_flow6,
                        'predict_flow5': predict_flow5,
                        'predict_flow4': predict_flow4,
                        'predict_flow3': predict_flow3,
                        'predict_flow2': predict_flow2,
                        'flow': flow
                    }

    def loss(self, flow, predictions):
        return


