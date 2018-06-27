import tensorflow.contrib.layers as tcl
import tensorflow as tf
import random
import os
import gzip
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tf_ops import *
import requests
from tensorflow.examples.tutorials.mnist import input_data

'''
def G(z,c):

   zc = tf.concat([z, c], axis=1)

   g_fc1 = tcl.fully_connected(zc, 1024, activation_fn=tf.nn.relu, scope='g_fc1')
   g_fc2 = tcl.fully_connected(g_fc1, 7*7*128, activation_fn=tf.nn.relu, scope='g_fc2')

   g_fc2_r = tf.reshape(g_fc2, [-1, 7, 7, 128])

   g_conv1 = tcl.conv2d_transpose(g_fc2_r, 64, 4, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), normalizer_fn=tcl.batch_norm, scope='g_conv1')
   
   g_conv2 = tcl.conv2d_transpose(g_conv1, 1, 4, stride=2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), normalizer_fn=tcl.batch_norm, scope='g_conv2')
   g_conv2 = tf.nn.sigmoid(g_conv2)

   print 'z:',z
   print 'g_fc1:',g_fc1
   print 'g_fc2:',g_fc2
   print 'g_fc2_r:',g_fc2_r
   print 'g_conv1:',g_conv1
   print 'g_conv2:',g_conv2
   print
   return g_conv2


def D(x,reuse=False):

   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = tcl.conv2d(x, 64, 4, stride=2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
      conv1 = lrelu(conv1, leak=0.1)
      
      conv2 = tcl.conv2d(conv1, 128, 4, stride=2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), normalizer_fn=tcl.batch_norm, scope='d_conv2')
      conv2 = lrelu(conv2, leak=0.1)

      conv2_flat = tcl.flatten(conv1)

      fc1 = tcl.fully_connected(conv2_flat, 1024, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), normalizer_fn=tcl.batch_norm, scope='d_fc1')
      fc1 = lrelu(fc1, leak=0.1)

      d_out = tcl.fully_connected(fc1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer(stddev=0.02), normalizer_fn=tcl.batch_norm, scope='d_out')

      q_1 = tcl.fully_connected(fc1, 128, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), normalizer_fn=tcl.batch_norm, scope='d_q1')
      q_1 = lrelu(q_1, leak=0.1)

      q_out = tcl.fully_connected(q_1, 10, activation_fn=tf.nn.softmax, weights_initializer=tf.random_normal_initializer(stddev=0.02), normalizer_fn=tcl.batch_norm, scope='d_qout')

   print 'x:',x
   print 'd_conv1:',conv1
   print 'd_conv2:',conv2
   print 'fc1:',fc1
   print 'd_out:',d_out
   print 'q_out:',q_out
   print
   return d_out, q_out
'''

def G(z, c):
   z = tf.concat([z,c], axis=1)
   fc1 = tcl.fully_connected(z, 256, weights_initializer=tcl.xavier_initializer(), activation_fn=tf.nn.relu, scope='g_fc1')
   fc2 = tcl.fully_connected(fc1, 784, weights_initializer=tcl.xavier_initializer(),activation_fn=tf.nn.sigmoid, scope='g_fc2')
   return fc2

def D(x, reuse=False):
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):
      fc1   = tcl.fully_connected(x, 128, weights_initializer=tcl.xavier_initializer(),activation_fn=tf.nn.relu, scope='d_fc1')
      fc2   = tcl.fully_connected(fc1, 1, weights_initializer=tcl.xavier_initializer(),activation_fn=tf.nn.sigmoid, scope='d_fc2')
      return fc2

def Q(x):
   fc1 = tcl.fully_connected(x, 128, weights_initializer=tcl.xavier_initializer(),activation_fn=tf.nn.relu, scope='q_fc1')
   fc2 = tcl.fully_connected(fc1, 10, weights_initializer=tcl.xavier_initializer(),activation_fn=tf.nn.softmax, scope='q_fc2')
   return fc2


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

if __name__ == '__main__':

   batch_size = 16
   mnist      = input_data.read_data_sets('../../MNIST_data', one_hot=True)

   try: os.makedirs('checkpoints/images/')
   except: pass

   global_step = tf.Variable(0, trainable=False, name='global_step')
   images      = tf.placeholder(tf.float32, [batch_size, 784], name='images')
   
   z = tf.placeholder(tf.float32, [batch_size, 16], name='z')
   c = tf.placeholder(tf.float32, [batch_size, 10], name='c')

   generated_images = G(z, c)

   D_real = D(images)
   D_fake = D(generated_images, reuse=True)
   q_fake = Q(generated_images)

   e = 1e-8
   errD = -tf.reduce_mean(tf.log(D_real+e)+tf.log(1-D_fake+e))
   errG = -tf.reduce_mean(tf.log(D_fake+e))

   cost = tf.reduce_mean(-tf.reduce_sum(tf.log(q_fake+e)*c,1))
   ent  = tf.reduce_mean(-tf.reduce_sum(tf.log(c+e)*c, 1))
   errQ = cost+ent

   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]
   q_vars = [var for var in t_vars if 'q_' in var.name]

   #G_train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(errG, var_list=g_vars, global_step=global_step)
   #D_train_op = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(errD, var_list=d_vars)
   G_train_op = tf.train.AdamOptimizer().minimize(errG, var_list=g_vars, global_step=global_step)
   D_train_op = tf.train.AdamOptimizer().minimize(errD, var_list=d_vars)
   Q_train_op = tf.train.AdamOptimizer().minimize(errQ, var_list=q_vars+g_vars)

   saver = tf.train.Saver(max_to_keep=1)

   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
   sess.run(init)

   # load previous checkpoint if there is one
   ckpt = tf.train.get_checkpoint_state('checkpoints/')
   if ckpt and ckpt.model_checkpoint_path:
      try: saver.restore(sess, ckpt.model_checkpoint_path)
      except: pass

   step = sess.run(global_step)
   num_train = 70000
   while True:
      step += 1

      epoch_num = step/(num_train/batch_size)

      batch_images,_ = mnist.train.next_batch(batch_size)
      
      batch_z = np.random.uniform(-1., 1., size=[batch_size, 16])
      idx = np.random.randint(0, 10)
      batch_c = np.zeros([batch_size, 10])
      batch_c[range(batch_size), idx] = 1
      
      _, d_loss = sess.run([D_train_op, errD], feed_dict={images:batch_images, z:batch_z, c:batch_c})
      _, g_loss = sess.run([G_train_op, errG], feed_dict={z:batch_z, c:batch_c})
      _, q_loss = sess.run([Q_train_op, errQ], feed_dict={z:batch_z, c:batch_c})

      if step%50==0:print 'epoch:',epoch_num,'step:',step,'G loss:',g_loss,' D loss:',d_loss,' Q loss:',q_loss

      if step%1000 == 0:
         print 'Saving model'
         saver.save(sess, 'checkpoints/checkpoint-', global_step=global_step)

         # generate some to write out
         batch_z = np.random.uniform(-1., 1., size=[batch_size, 16])
         batch_c = np.zeros([batch_size, 10])
         batch_c[range(batch_size), idx] = 1
         
         gen_imgs = np.asarray(sess.run([generated_images], feed_dict={z:batch_z, c:batch_c}))[0]
         gen_imgs = np.squeeze(gen_imgs)

         fig = plot(gen_imgs)
         plt.savefig('checkpoints/images/{}.png'.format(str(step).zfill(3)), bbox_inches='tight')
         plt.close(fig)

