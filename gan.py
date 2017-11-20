import tensorflow.contrib.layers as tcl
import tensorflow as tf
import random
import os
import gzip
import time
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tf_ops import *


def G(z,c):

   zc = tf.concat([z, c], axis=1)

   g_fc1 = tcl.fully_connected(zc, 1024, activation_fn=tf.nn.relu, scope='g_fc1')
   g_fc2 = tcl.fully_connected(g_fc1, 7*7*128, activation_fn=tf.nn.relu, scope='g_fc2')

   g_fc2_r = tf.reshape(g_fc2, [-1, 7, 7, 128])

   #g_conv1 = upconv2d(g_fc2_r, 64, new_height=14, new_width=14, kernel_size=4, name='g_conv1')
   g_conv1 = tcl.conv2d_transpose(g_fc2_r, 64, 4, stride=2, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), normalizer_fn=tcl.batch_norm, scope='g_conv1')
   #g_conv1 = relu(g_conv1)
   
   g_conv2 = tcl.conv2d_transpose(g_conv1, 1, 4, stride=2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), normalizer_fn=tcl.batch_norm, scope='g_conv2')
   #g_conv2 = upconv2d(g_conv1, 1, new_height=28, new_width=28, kernel_size=4, name='g_conv2')
   #g_conv2 = tf.nn.sigmoid(g_conv2)

   print 'z:',z
   print 'g_fc1:',g_fc1
   print 'g_fc2:',g_fc2
   print 'g_fc2_r:',g_fc2_r
   print 'g_conv1:',g_conv1
   print 'g_conv2:',g_conv2
   return g_conv2


def D(x,reuse=False):

   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = tcl.conv2d(x, 64, 4, stride=2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), normalizer_fn=None, scope='d_conv1')
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
   return d_out, q_out

def train(mnist_train):
   with tf.Graph().as_default():
     
      batch_size = 64

      # placeholder to keep track of the global step
      global_step = tf.Variable(0, trainable=False, name='global_step')
      
      # placeholder for mnist images
      images = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name='images')
      
      # placeholder for the latent z vector
      z = tf.placeholder(tf.float32, [batch_size, 90], name='z')
      c = tf.placeholder(tf.float32, [batch_size, 10], name='c')

      # generate an image from noise prior z
      generated_images = G(z, c)

      D_real, q_real = D(images)
      D_fake, q_fake = D(generated_images, reuse=True)

      e = 1e-8
      errD = tf.reduce_mean(-(tf.log(D_real+e)+tf.log(1-D_fake+e)))

      # instead of minimizing (1-D(G(z)), maximize D(G(z))
      errG = tf.reduce_mean(-tf.log(D_fake+e))

      cost = tf.reduce_mean(-tf.reduce_sum(tf.log(q_fake+e)*c,1))
      ent  = tf.reduce_mean(-tf.reduce_sum(tf.log(c+e)*c, 1))
      errQ = cost+ent

      # get all trainable variables, and split by network G and network D
      t_vars = tf.trainable_variables()
      d_vars = [var for var in t_vars if 'd_' in var.name]
      g_vars = [var for var in t_vars if 'g_' in var.name]

      # training operators for G and D
      G_train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(errG, var_list=g_vars, global_step=global_step)
      D_train_op = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(errD, var_list=d_vars)
      Q_train_op = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(errQ, var_list=d_vars)

      saver = tf.train.Saver(max_to_keep=1)
   
      init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
      sess.run(init)

      # tensorboard summaries
      try: tf.summary.scalar('d_loss', tf.reduce_mean(errD))
      except:pass
      try: tf.summary.scalar('g_loss', tf.reduce_mean(errG))
      except:pass

      # write out logs for tensorboard to the checkpointSdir
      summary_writer = tf.summary.FileWriter('checkpoints/gan/logs/', graph=tf.get_default_graph())

      # load previous checkpoint if there is one
      ckpt = tf.train.get_checkpoint_state('checkpoints/gan/')
      if ckpt and ckpt.model_checkpoint_path:
         try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Model restored'
         except:
            print 'Could not restore model'
            pass

      merged_summary_op = tf.summary.merge_all()
      # training loop
      step = sess.run(global_step)
      num_train = len(mnist_train)
      while True:
         s = time.time()
         step += 1

         epoch_num = step/(num_train/batch_size)

         # get random images from the training set
         batch_images = random.sample(mnist_train, batch_size)
         
         # generate z from a normal/uniform distribution between [-1, 1] of length 100
         batch_z = np.random.uniform(-1.0, 1.0, size=[batch_size, 90]).astype(np.float32)
         batch_c = np.random.multinomial(1, 10*[0.1], size=[batch_size]).astype(np.float32)
         
         _, d_loss = sess.run([D_train_op, errD], feed_dict={images:batch_images, z:batch_z, c:batch_c})
         _, q_loss = sess.run([Q_train_op, errQ], feed_dict={images:batch_images, z:batch_z, c:batch_c})
         _, g_loss = sess.run([G_train_op, errG], feed_dict={z:batch_z, c:batch_c})
         
         if step%10==0:print 'epoch:',epoch_num,'step:',step,'G loss:',g_loss,' D loss:',d_loss,' Q loss:',q_loss

         if step%1000 == 0:
            print
            print 'Saving model'
            print
            saver.save(sess, 'checkpoints/gan/checkpoint-', global_step=global_step)

            # generate some to write out
            batch_z = np.random.uniform(-1.0, 1.0, size=[batch_size, 90]).astype(np.float32)
            batch_c = np.random.multinomial(1, 10*[0.1], size=[batch_size]).astype(np.float32)
            
            gen_imgs = np.asarray(sess.run([generated_images], feed_dict={z:batch_z, c:batch_c}))[0]
            gen_imgs = np.squeeze(gen_imgs)
            count = 0
            #canvas = np.ones(((3*28), (5*28)+50), dtype=np.uint8)
            #start_x = 8
            #start_y = 8

            for img in gen_imgs:

               plt.imsave('checkpoints/gan/images/0000'+str(step)+'_'+str(count)+'.png', img, cmap=plt.cm.gray)

               count += 1
               if count == 4: break
               '''
               plt.imsave('checkpoints/gan/images/image'+str(step)+'.png', img, cmap=plt.cm.gray)
               img = np.reshape(img, [28, 28])
               end_x = start_x+28
               end_y = start_y+28

               canvas[start_y:end_y, start_x:end_x] = img

               if c == 4:
                  start_x = 8
                  start_y = end_y + 8
               else:
                  start_x = end_x + 8
               if c == 9:
                  break
               c+=1
               '''
            #plt.imsave('checkpoints/gan/images/0000'+str(step)+'.png', canvas)#, cmap=plt.cm.gray)

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

   try: os.mkdir('checkpoints/')
   except: pass
   try: os.mkdir('checkpoints/gan/')
   except: pass
   try: os.mkdir('checkpoints/gan/images/')
   except: pass
   try: os.mkdir('checkpoints/gan/logs/')
   except: pass
   
   url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'

   # check if it's already downloaded
   if not os.path.isfile('mnist.pkl.gz'):
      print 'Downloading mnist...'
      with open('mnist.pkl.gz', 'wb') as f:
         r = requests.get(url)
         if r.status_code == 200:
            f.write(r.content)
         else:
            print 'Could not connect to ', url

   print 'opening mnist'
   f = gzip.open('mnist.pkl.gz', 'rb')
   train_set, val_set, test_set = pickle.load(f)

   mnist_train = []

   # we will be using all splits. This goes through and adds a dimension
   # to the images and adds them to a training set
   for t,l in zip(*train_set):
      mnist_train.append(np.reshape(t, (28, 28, 1)))
   for t,l in zip(*val_set):
      mnist_train.append(np.reshape(t, (28, 28, 1)))
   for t,l in zip(*test_set):
      mnist_train.append(np.reshape(t, (28, 28, 1)))

   mnist_train = np.asarray(mnist_train)


   train(mnist_train)
