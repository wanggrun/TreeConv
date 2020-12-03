# -*- coding: utf-8 -*-
# File: conv2d.py

from tensorpack.compat import tfv1 as tf  # this should be avoided first in model code

from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.utils.argtools import get_data_format, shape2d, shape4d, log_once
from tensorpack.models.common import VariableHolder, layer_register
from tensorpack.models.tflayer import convert_to_tflayer_args, rename_get_variable

__all__ = ['Conv']


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def Conv(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        split=1,
        norm = False):
    """
    Similar to `tf.layers.Conv2D`, but with some differences:
    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'.
    3. Support 'split' argument to do group convolution.
    Variable Names:
    * ``W``: weights
    * ``b``: bias
    """
    if kernel_initializer is None:
        if get_tf_version_tuple() <= (1, 12):
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)  # deprecated
        else:
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal')
    dilation_rate = shape2d(dilation_rate)

    if True:
        # group conv implementation
        data_format = get_data_format(data_format, keras_mode=False)
        in_shape = inputs.get_shape().as_list()
        channel_axis = 3 if data_format == 'NHWC' else 1
        in_channel = in_shape[channel_axis]
        assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
        assert in_channel % split == 0

        assert kernel_regularizer is None and bias_regularizer is None and activity_regularizer is None, \
            "Not supported by group conv or dilated conv!"

        out_channel = filters
        assert out_channel % split == 0
        assert dilation_rate == [1, 1] or get_tf_version_tuple() >= (1, 5), 'TF>=1.5 required for dilated conv.'

        kernel_shape = shape2d(kernel_size)
        filter_shape = kernel_shape + [in_channel // split, out_channel]
        stride = shape4d(strides, data_format=data_format)

        kwargs = {"data_format": data_format}
        if get_tf_version_tuple() >= (1, 5):
            kwargs['dilations'] = shape4d(dilation_rate, data_format=data_format)

        # matching input dtype (ex. tf.float16) since the default dtype of variable if tf.float32
        inputs_dtype = inputs.dtype
        W = tf.get_variable(
            'parseweigth', filter_shape, dtype=inputs_dtype, initializer=kernel_initializer)
        if norm:
            use_bias = False
            W = tf.reshape(W, kernel_shape + [4, in_channel//4, out_channel])
            W = tf.nn.softmax(W, 2)
            W = tf.reshape(W, filter_shape)
        #dynamics = tf.reduce_mean(inputs, 0)
        #dynamics = tf.transpose(dynamics, [1,2,0])
        #dynamics = tf.image.resize_images(dynamics, kernel_shape)
        #dynamics = tf.expand_dims(dynamics, -1)
        #W = W  +  0.001 * dynamics #tf.random_normal(shape = tf.shape(W), mean = 0.0, stddev = 0.012, dtype = tf.float32)        
   
        #W = W *tf.random_uniform(shape=W.get_shape().as_list(), minval=0., maxval=2.)

        if use_bias:
            b = tf.get_variable('parsebias', [out_channel], dtype=inputs_dtype, initializer=bias_initializer)

        if split == 1:
            conv = tf.nn.conv2d(inputs, W, stride, padding.upper(), **kwargs)
        else:
            try:
                conv = tf.nn.conv2d(inputs, W, stride, padding.upper(), **kwargs)
            except ValueError:
                log_once("CUDNN group convolution support is only available with "
                         "https://github.com/tensorflow/tensorflow/pull/25818 . "
                         "Will fall back to a loop-based slow implementation instead!", 'warn')

        ret = tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv
        if activation is not None:
            ret = activation(ret)
        ret = tf.identity(ret, name='output')

        ret.variables = VariableHolder(W=W)
        if use_bias:
            ret.variables.b = b
    return ret




