from .flownet_c import FlowNetC
from .flownet_s import FlowNetS
from .flow_warp import flow_warp
import tensorflow as tf


class FlowNetCS():

    def __init__(self):
        self.net_c = FlowNetC()
        self.net_s = FlowNetS()
    def model(self, inputs,trainable=True,reuse=False):
        with tf.variable_scope('FlowNetCS',reuse=reuse):
            # Forward pass through FlowNetC with weights frozen
            net_c_predictions = self.net_c.model(inputs,trainable=False)
            flow = net_c_predictions['flow']

            warped = flow_warp(inputs['input_b'], flow)

            brightness_error = inputs['input_a'] - warped
            brightness_error = tf.square(brightness_error)
            brightness_error = tf.reduce_sum(brightness_error, keep_dims=True, axis=3)
            brightness_error = tf.sqrt(brightness_error)

            # Gather all inputs to FlowNetS
            inputs_to_s = {
                'input_a': inputs['input_a'],
                'input_b': inputs['input_b'],
                'warped': warped,
                'flow': flow * 0.05,
                'brightness_error': brightness_error,
            }
            return self.net_s.model(inputs_to_s,trainable=trainable)

    def loss(self, flow, predictions):
        return

