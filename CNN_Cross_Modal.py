# L1 - cross entropy - corresponding classification - appending - BIG vectors

from alexnet import AlexNet
from datetime import datetime
from tensorflow.contrib.data import Iterator
from scipy import ndimage, misc
from skimage import io, img_as_float, transform
from datagenerator import ImageDataGenerator

import tensorflow as tf
import numpy as np
import pdb, sys, os

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var) 
        tf.summary.scalar('eucl Norm', tf.norm(var))

def get_weight(shape, name, trainable=True):
    weights = tf.get_variable(name, shape, trainable=trainable)

    return weights

def get_bias(num_nodes, name, trainable=True):
    bias = tf.get_variable(name, [num_nodes], trainable=trainable)

    return bias

def get_dense_layer(input_data, num_in, num_out, name, relu=True):

    with tf.name_scope(name):
        # Create tf variables for the weights and biases
        with tf.name_scope('weights'):
            weights = get_weight([num_in, num_out], name+'_weights')
            variable_summaries(weights)
    
        with tf.name_scope('bias'):
            bias = get_bias(num_out, name+'_bias')
            variable_summaries(bias)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(input_data, weights, bias)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

def get_dropout_layer(input_data, keep_prob, name):
    output_data = tf.nn.dropout(input_data, keep_prob=keep_prob, name=name)

    return output_data

# custom loss functions
def euclidean_loss(mat1, mat2):
    with tf.variable_scope("euclidean_loss"):
        diff = tf.subtract(mat1, mat2)
        diff_squared = tf.square(diff)
        error = tf.reduce_sum(diff_squared)

        return error

def mmd_loss(mat1, mat2):
    with tf.variable_scope("mmd_loss"):
        diff = tf.subtract(mat1, mat2)
        mmd = tf.reduce_mean(diff, 0)
        error = tf.norm(mmd, ord='euclidean')

        return error    

def cross_entropy_loss(mat1, mat2):
    with tf.variable_scope("cross_entropy_loss"):
        y = mat1
        y_ = tf.nn.softmax(mat2)
        y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
        cross_entropy = -tf.reduce_sum(y * tf.log(y_clipped))

        return cross_entropy

def sigmoid_loss(mat1, mat2):
    with tf.variable_scope("sigmoid_loss"):
        y = mat1
        y_ = tf.nn.sigmoid(mat2)
        y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
        sigmoid_loss = -tf.reduce_sum(y * tf.log(y_clipped) + (1-y) * tf.log(1-y_clipped))

        return sigmoid_loss

def batch_norm_wrapper(inputs, is_training=True, decay = 0.999):
    epsilon = 1e-3
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

# variables 
batch_size = None
num_classes = 20

source_image_size = [227, 227, 3]
source_text_size = [332]
source_imglabel_size = [num_classes]
source_txtlabel_size = [num_classes]

source_image_input = tf.placeholder(tf.float32, [batch_size] + source_image_size, 'source_image')
source_text_input = tf.placeholder(tf.float32, [batch_size] + source_text_size, 'source_text')
source_imglabel_input = tf.placeholder(tf.float32, [batch_size] + source_imglabel_size, 'source_imglabel')
source_txtlabel_input = tf.placeholder(tf.float32, [batch_size] + source_txtlabel_size, 'source_txtlabel')
source_corres_input = tf.placeholder(tf.float32, [batch_size] + [2], 'source_corres')

# source image
train_layers = ['fc8', 'fc7', 'fc6']

keep_prob = tf.placeholder(tf.float32)
model = AlexNet(source_image_input, keep_prob, num_classes, train_layers)

alex_output = model.output

SI_dense1_num = 2048
SI_dense1 = get_dense_layer(alex_output, 4096, SI_dense1_num, relu=False, name='SI_dense1')

SI_dense2_num = 1024
SI_dense2 = get_dense_layer(SI_dense1, SI_dense1_num, SI_dense2_num, relu=False, name='SI_dense2')

SI_hidden = SI_dense2

# source text
ST_dense1_num = 2048
ST_dense1 = get_dense_layer(source_text_input, source_text_size[0], ST_dense1_num, relu=True, name='ST_dense1')

ST_dense2_num = 1024
ST_dense2 = get_dense_layer(ST_dense1, ST_dense1_num, ST_dense2_num, relu=True, name='ST_dense2')

ST_hidden = ST_dense2

# common layers

# common layer 1
C_dense1_num = 1024
C_dense1_weights = get_weight([1024, C_dense1_num], 'C_dense1_weight')
C_dense1_bias = get_bias(C_dense1_num, 'C_dense1_bias')

variable_summaries(C_dense1_weights)
variable_summaries(C_dense1_bias)

CSI_dense1 = tf.nn.xw_plus_b(SI_hidden, C_dense1_weights, C_dense1_bias, name='CSI_dense1')
CSI_dense1 = tf.nn.relu(CSI_dense1)

CST_dense1 = tf.nn.xw_plus_b(ST_hidden, C_dense1_weights, C_dense1_bias, name='CST_dense1')
CST_dense1 = tf.nn.relu(CST_dense1)

# common layer 2
C_dense2_num = 512
C_dense2_weights = get_weight([C_dense1_num, C_dense2_num], 'C_dense2_weight')
C_dense2_bias = get_bias(C_dense2_num, 'C_dense2_bias')

variable_summaries(C_dense2_weights)
variable_summaries(C_dense2_bias)

CSI_dense2 = tf.nn.xw_plus_b(CSI_dense1, C_dense2_weights, C_dense2_bias, name='CSI_dense2')
CST_dense2 = tf.nn.xw_plus_b(CST_dense1, C_dense2_weights, C_dense2_bias, name='CST_dense2')

# common layer 3
C_dense3_num = num_classes
C_dense3_weights = get_weight([C_dense2_num, C_dense3_num], 'C_dense3_weight')
C_dense3_bias = get_bias(C_dense3_num, 'C_dense3_bias')

variable_summaries(C_dense3_weights)
variable_summaries(C_dense3_bias)

CSI_dense3 = tf.nn.xw_plus_b(CSI_dense2, C_dense3_weights, C_dense3_bias, name='CSI_dense3')
CST_dense3 = tf.nn.xw_plus_b(CST_dense2, C_dense3_weights, C_dense3_bias, name='CST_dense3')

CSI_hidden1 = CSI_dense1
CST_hidden1 = CST_dense1

CSI_hidden2 = CSI_dense2
CST_hidden2 = CST_dense2

CSI_hidden3 = CSI_dense3
CST_hidden3 = CST_dense3

# errors and optimizers
learning_rate = 0.001

# Image L3
source_image_l3_loss = cross_entropy_loss(source_imglabel_input, CSI_hidden3)

source_image_pred = tf.equal(tf.argmax(CSI_hidden3, 1), tf.argmax(source_imglabel_input, 1))
source_image_l3_accuracy = tf.reduce_mean(tf.cast(source_image_pred, tf.float32))

optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss=source_image_l3_loss)
# optimizer1 = tf.train.AdadeltaOptimizer(1., 0.95, 1e-6).minimize(loss=source_image_l3_loss)
# optimizer1 = tf.train.AdadeltaOptimizer(0.001, 0.95, 1e-6).minimize(loss=source_image_l3_loss)

# Text L3
source_text_l3_loss = cross_entropy_loss(source_txtlabel_input, CST_hidden3)

source_text_pred = tf.equal(tf.argmax(tf.nn.softmax(CST_hidden3), 1), tf.argmax(source_txtlabel_input, 1))
source_text_l3_accuracy = tf.reduce_mean(tf.cast(source_text_pred, tf.float32))

optimizer2 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=source_text_l3_loss)

# Source L1
concat = tf.concat([CSI_hidden1, CST_hidden1], axis=1)

# corres_num = 2048
# corres_weights = get_weight([2048, corres_num], 'corres_weight')
# corres_bias = get_bias(corres_num, 'corres_bias')
# corres_dense = tf.nn.xw_plus_b(concat, corres_weights, corres_bias, name='corres_dense')

corres_dense1_num = 512
corres_dense1 = get_dense_layer(concat, 2048, corres_dense1_num, relu=True, name='corres_dense1')

corres_dense2_num = 2
corres_dense2 = get_dense_layer(corres_dense1, corres_dense1_num, corres_dense2_num, relu=False, name='corres_dense2')

source_l1_loss = 0.001 * cross_entropy_loss(source_corres_input, corres_dense2)
source_l1_pred = tf.equal(tf.argmax(corres_dense2, 1), tf.argmax(source_corres_input, 1))
source_l1_accuracy = tf.reduce_mean(tf.cast(source_l1_pred, tf.float32))

optimizer3 = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(loss=source_l1_loss)

# Summary OPs
summ_image_l3_loss = tf.summary.scalar('src_image_l3_loss_cross_entropy', source_image_l3_loss)
summ_text_l3_loss = tf.summary.scalar('src_text_l3_loss_cross_entropy', source_text_l3_loss)
summ_image_acc = tf.summary.scalar('src_image_acc', source_image_l3_accuracy)
summ_text_acc = tf.summary.scalar('src_text_acc', source_text_l3_accuracy)
summ_l1_loss = tf.summary.scalar('src_l1_loss_softmax_ce', source_l1_loss)
summ_l1_acc = tf.summary.scalar('src_l1_acc', source_l1_accuracy)
merged = tf.summary.merge_all()

# dataset loader
batch_size = 20
dropout_rate = 0.5
train_file = 'tiny_pascal_train.txt'
val_file = 'tiny_pascal_test.txt'

tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)

val_data = ImageDataGenerator(val_file,
                                 mode='inference',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=False)

iterator = Iterator.from_structure(tr_data.data.output_types,
                                   tr_data.data.output_shapes)
next_batch = iterator.get_next()

training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# training the neural net
train_epoch = 20
test_epoch = 1

# print tf.trainable_variables()

expt_name = "L-123_sgdi_adamt_crossentropyl1_mk"

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('./summary_stats/' + expt_name +'train/', sess.graph)
    test_writer = tf.summary.FileWriter('./summary_stats/' + expt_name  +'test/')

    sess.run(tf.global_variables_initializer())
    model.load_initial_weights(sess)

    train_l3_ctr = 0
    train_l1_ctr = 0
    
    test_l3_ctr = 0
    test_l1_ctr = 0
    
    for epoch in range(train_epoch):
	print "Epoch number:", epoch
	print "Initialising training dataset generator for L3"
	
	sess.run(training_init_op)
        while True:
            try:
            	source_imagename_batch, source_textname_batch, source_image_batch, source_text_batch, source_imglabel_batch, source_txtlabel_batch, source_corres_batch = sess.run(next_batch)
	    except Exception, e:
		break
                # continue 
            
            num_instances = source_image_batch.shape[0] 
            
            if num_instances < batch_size:
                break
            
            sess.run(optimizer1, feed_dict={source_image_input: source_image_batch, source_imglabel_input: source_imglabel_batch, keep_prob: dropout_rate})
            acc = sess.run(source_image_l3_accuracy, feed_dict={source_image_input: source_image_batch, source_imglabel_input: source_imglabel_batch, keep_prob: dropout_rate})
            los = sess.run(source_image_l3_loss, feed_dict={source_image_input: source_image_batch, source_imglabel_input: source_imglabel_batch, keep_prob: dropout_rate}) 
            
            print "train Source Image Loss:", los, "Accuracy:", acc
           
            sess.run(optimizer2, feed_dict={source_text_input: source_text_batch, source_txtlabel_input: source_txtlabel_batch})
            acc = sess.run(source_text_l3_accuracy, feed_dict={source_text_input: source_text_batch, source_txtlabel_input: source_txtlabel_batch})
            los = sess.run(source_text_l3_loss, feed_dict={source_text_input: source_text_batch, source_txtlabel_input: source_txtlabel_batch})
             
            print "train Source Text Loss:", los, "Accuracy:", acc

            sess.run(optimizer3, feed_dict={source_text_input: source_text_batch, source_image_input: source_image_batch, source_corres_input: source_corres_batch, keep_prob: dropout_rate})
            los = sess.run(source_l1_loss, feed_dict={source_text_input: source_text_batch, source_image_input: source_image_batch, source_corres_input: source_corres_batch, keep_prob: dropout_rate})
            acc = sess.run(source_l1_accuracy, feed_dict={source_text_input: source_text_batch, source_image_input: source_image_batch, source_corres_input: source_corres_batch, keep_prob: dropout_rate}) 
            
            print "train Source L1 Loss:", los, "Accuracy:", acc
            
            print
            print            
            
            summary = sess.run(merged, 
                               feed_dict = {source_image_input: source_image_batch,
                                            source_imglabel_input: source_imglabel_batch,
                                            source_text_input: source_text_batch,
                                            source_txtlabel_input: source_txtlabel_batch,
                                            source_corres_input: source_corres_batch,
                                            keep_prob: dropout_rate })
            
            train_writer.add_summary(summary, train_l1_ctr)
            train_l1_ctr += 1 

	print "Performing validation"
        print "Initialising test dataset generator for L3"
	sess.run(validation_init_op)
        while True:
            try:
            	source_imagename_batch, source_textname_batch, source_image_batch, source_text_batch, source_imglabel_batch, source_txtlabel_batch, source_corres_batch = sess.run(next_batch)
	    except Exception, e:
		break
                # continue 

            num_instances = source_image_batch.shape[0] 
            if num_instances < batch_size:
        	break
            
            acc = sess.run(source_image_l3_accuracy, feed_dict={source_image_input: source_image_batch, source_imglabel_input: source_imglabel_batch, keep_prob: dropout_rate})
            los = sess.run(source_image_l3_loss, feed_dict={source_image_input: source_image_batch, source_imglabel_input: source_imglabel_batch,keep_prob: dropout_rate}) 

            print "test Source Image Loss:", los, "Accuracy:", acc

            acc = sess.run(source_text_l3_accuracy, feed_dict={source_text_input: source_text_batch, source_txtlabel_input: source_txtlabel_batch, keep_prob: dropout_rate})
            los = sess.run(source_text_l3_loss, feed_dict={source_text_input: source_text_batch, source_txtlabel_input: source_txtlabel_batch, keep_prob: dropout_rate}) 

            print "test Source text Loss:", los, "Accuracy:", acc

            los = sess.run(source_l1_loss, feed_dict={source_text_input: source_text_batch, source_image_input: source_image_batch, source_corres_input: source_corres_batch, keep_prob: dropout_rate})
            acc = sess.run(source_l1_accuracy, feed_dict={source_text_input: source_text_batch, source_image_input: source_image_batch, source_corres_input: source_corres_batch, keep_prob: dropout_rate}) 

            print "test Source L1 Loss:", los, "Accuracy:", acc

            print
            print
            # print "test result check"

            summary = sess.run(merged, 
                               feed_dict = {source_image_input: source_image_batch,
                                            source_imglabel_input: source_imglabel_batch,
                                            source_text_input: source_text_batch,
                                            source_txtlabel_input: source_txtlabel_batch,
                                            source_corres_input: source_corres_batch,
                                            keep_prob: dropout_rate })

            test_writer.add_summary(summary, test_l1_ctr)
            test_l1_ctr += 1 

    # write hidden representations
    sess.run(training_init_op)
    
    f1 = open('image_hidden.txt', 'w')
    f2 = open('text_hidden.txt', 'w')
    
    while True:
       try:
           source_imagename_batch, source_textname_batch, source_image_batch, source_text_batch, source_imglabel_batch, source_txtlabel_batch, source_corres_batch = sess.run(next_batch)
       except:
           break
           # continue

       # print source_imagename_batch
       # print source_textname_batch
       # print source_image_batch
       # print source_text_batch
       # print source_label_batch

       num_instances = source_image_batch.shape[0] 
       if num_instances < batch_size:
           break

       image_hidden = sess.run(CSI_hidden3, feed_dict={source_image_input: source_image_batch,
                                                     keep_prob: dropout_rate})

       text_hidden = sess.run(CST_hidden3, feed_dict={source_text_input: source_text_batch,
                                                    keep_prob: dropout_rate})
          
       for i in range(batch_size):
           f1.write(str(source_imagename_batch[i])+'|')
	   f1.write(' '.join(map(str, image_hidden[i]))+'\n')

           f2.write(str(source_textname_batch[i])+'|')
           f2.write(' '.join(map(str, text_hidden[i]))+'\n')
	
    f1.close()
    f2.close()
