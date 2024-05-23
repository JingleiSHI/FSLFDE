from .flownet_cs import FlowNetCS
from .flownet_s import FlowNetS
from .flow_warp import flow_warp
import tensorflow as tf


class FlowNetCSS():
    def __init__(self):
        self.net_cs = FlowNetCS()
        self.net_s = FlowNetS()
    def model(self, inputs,trainable=True,reuse=False):
        with tf.variable_scope('FlowNetCSS',reuse=reuse):
            # Forward pass through FlowNetCS with weights frozen
            net_cs_predictions = self.net_cs.model(inputs,trainable=False)

            flow = net_cs_predictions['flow']
            # Perform flow warping (to move image B closer to image A based on flow prediction)
            warped = flow_warp(inputs['input_b'], flow)

            # Compute brightness error: sqrt(sum (input_a - warped)^2 over channels)
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

