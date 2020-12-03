# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf

from tensorpack.models import BatchNorm, BNReLU, Conv2D, FullyConnected, GlobalAvgPooling, MaxPooling, AvgPooling, LayerNorm
from tensorpack.tfutils.argscope import argscope, get_arg_scope
import numpy as np

from  conv import  Conv


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format in ['NCHW', 'channels_first'] else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)


# ----------------- pre-activation resnet ----------------------
def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut


def preact_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


def preact_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def preact_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l
# ----------------- pre-activation resnet ----------------------


def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def resnet_bottleneck_ori(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)

def sigmoid(x, alpha):
  out = 1. / (1 + tf.exp(- alpha * x))
  return out 


def parse(l, ch_out, stride, channel_wise = True):
    shape = l.get_shape().as_list()
    l_ori = l
    parse1 = tf.nn.sigmoid(Conv('parse1', l, 1, 1, strides=1))
    l1_left = l * parse1
    l = l - l1_left
    
    parse2 = tf.nn.sigmoid(Conv('parse2', l, 1, 1, strides=1))
    l2_left = l * parse2
    l = l - l2_left
  
    parse3 = tf.nn.sigmoid(Conv('parse3', l, 1, 1, strides=1))
    l3_left = l * parse3
    l3_right = l - l3_left

    l1_left = tf.keras.backend.resize_images(Conv2D('l1_left',  AvgPooling('pool', l1_left, pool_size=8, strides=8, padding='VALID'), 
             1*ch_out//4,  3 if shape[1]//8 > 2  else 1, strides=1), 8//stride, 8//stride, 'channels_last' )
    l2_left = tf.keras.backend.resize_images(Conv2D('l2_left', AvgPooling('pool', l2_left, pool_size=4, strides=4, padding='VALID'),
             1*ch_out//4, 3 if shape[1]//4 > 2  else 1, strides=1), 4//stride, 4//stride, 'channels_last')
    l3_left = tf.keras.backend.resize_images(Conv2D('l3_left',  AvgPooling('pool', l3_left, pool_size=2, strides=2, padding='VALID'),
             1*ch_out//4, 3 if shape[1]//2 > 2 else 1, strides=1), 2//stride, 2//stride, 'channels_last')
    l3_right = Conv2D('l3_right',  l3_right,
             1*ch_out//4, 3 if shape[1] > 2 else 1, strides=stride)
    
    l_ori = Conv2D('l_ori', l_ori, ch_out//4, 3, strides=stride, activation = BNReLU)
    
    l = tf.concat([tf.nn.sigmoid(BatchNorm('bn1', l1_left))*l_ori, tf.nn.sigmoid(BatchNorm('bn2',l2_left))*l_ori, 
          tf.nn.sigmoid(BatchNorm('bn3', l3_left))*l_ori, tf.nn.sigmoid(BatchNorm('bn4', l3_right))*l_ori], -1)
    return l


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=BNReLU)
    l = parse(l, ch_out, stride = stride)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def se_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, activation=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, activation=tf.nn.sigmoid)
    data_format = get_arg_scope()['Conv2D']['data_format']
    ch_ax = 1 if data_format in ['NCHW', 'channels_first'] else 3
    shape = [-1, 1, 1, 1]
    shape[ch_ax] = ch_out * 4
    l = l * tf.reshape(squeeze, shape)
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def resnext32x4d_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out * 2, 1, strides=1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out * 2, 3, strides=stride, activation=BNReLU, split=32)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def resnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func):
    with argscope(Conv2D, use_bias=False,
                  kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        # Note that TF pads the image by [2, 3] instead of [3, 2].
        # Similar things happen in later stride=2 layers as well.
        l = Conv2D('conv0', image, 64, 3, strides=2, activation=BNReLU)
        l = Conv2D('pool0', l, 64, 3, strides=2, activation=BNReLU)
        l = group_func('group0', l, block_func, 64, num_blocks[0], 1)#56
        shape = l.get_shape().as_list()
        l = tf.image.resize_images(l, [shape[1]//7*8, shape[2]//7*8], method=0)
        l = group_func('group1', l, block_func, 128, num_blocks[1], 2)#28
        l = group_func('group2', l, block_func, 256, num_blocks[2], 2)#14
        l = group_func('group3', l, block_func, 512, num_blocks[3], 2)#7
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return logits
