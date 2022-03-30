# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt


def variable_summaries(name, var, with_max_min=False):
    """ Tensor summaries for TensorBoard visualization """

    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)

        if with_max_min == True:
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))

def
if __name__ == '__main__':
    pass
