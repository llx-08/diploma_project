import os
import tensorflow as tf

LOGDIR = './mnist'

mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + 'data', one_hot=True)


def conv_layer(input, size_in, size_out):
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME')
    act = tf.nn.relu(conv + b)

    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def fc_layer(input, size_in, size_out):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    act = tf.nn.relu(tf.matmul(input, w) + b)

    return act


def mnist_model(learning_rate, use_two_conv, use_two_fc, hparam):
    tf.reset_default_graph()
    sess = tf.Session()

    # setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    y = tf.placeholder(tf.float32, shape=[None, 10])

    if use_two_conv:
        conv1 = conv_layer(x_image, 1, 32)
        conv_out = conv_layer(conv1, 32, 64)

    else:
        conv1 = conv_layer(x_image, 1, 64)
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    if use_two_fc:
        fc1 = fc_layer(flattened, 7 * 7 * 64, 1024)
        embedding_input = fc1
        embedding_size = 1024
        logits = fc_layer(fc1, 1024, 10)

    else:
        embedding_input = flattened
        embedding_size = 7 * 7 * 64
        logits = fc_layer(flattened, 7 * 7 * 64, 10)

    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    emdedding = tf.Variable(tf.zeros([1024, embedding_size]))
    assignment = emdedding.assign(embedding_input)

    sess.run(tf.global_variables_initializer())
    # 保存路径
    tenboard_dir = './tensorboard/test1/'

    # 指定一个文件用来保存图
    writer = tf.summary.FileWriter(tenboard_dir + hparam)
    # 把图add进去
    writer.add_graph(sess.graph)

    for i in range(2001):
        batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})


def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = 'conv=2' if use_two_conv else 'conv=1'
    fc_param = 'fc=2' if use_two_fc else 'fc=1'
    return 'lr_%.0E,%s,%s' % (learning_rate, conv_param, fc_param)


def main():
    # You can try adding some more learning rates
    for learning_rate in [1E-4]:

        # Include 'False' as a value to try different model architectures.
        for use_two_fc in [True]:
            for use_two_conv in [True]:
                # Construct a hyperparameter string for each one(example: 'lr_1E-3,fc=2,conv=2')
                hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                print('Starting run for %s' % hparam)

                # Actually run with the new settings
                mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)


if __name__ == '__main__':
    main()