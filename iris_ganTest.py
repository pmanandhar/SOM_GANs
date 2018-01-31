import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import load_digits as load_iris #load_breast_cancer #load_wine #load_boston load_linnerud load_diabetes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from random import sample
from numpy.random import randn,rand
from mpl_toolkits.mplot3d import Axes3D



iter=60000
iterfig=10000
mb_size = 1000
hdunits=1600
global Z_dim
global G_log_prob

batchsize=50
zbatchsize=batchsize
data1 = load_iris().data#input_data.read_data_sets('../../MNIST_data', one_hot=True)
data=data1[:,[0,1,2,3]]
dataCols=data.shape[1]
trainy= load_iris().target
Z_dim=500
#drop_out=0.8

def getGenWeight1():
	global G_log_prob
	return G_log_prob

def getGenWeight2():
	global G_W2
	return G_W2

def xav_init(size):
	in_dim = size[0]
	xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
	return tf.random_normal(shape=size, stddev=xavier_stddev)

def get_weights():
        return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('w1:0')]
X = tf.placeholder(tf.float32, shape=[None, dataCols])

D_W1 = tf.Variable(xav_init([dataCols, hdunits]))
D_b1 = tf.Variable(tf.zeros(shape=[hdunits]))

D_W2 = tf.Variable(xav_init([hdunits, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xav_init([Z_dim, hdunits]), name='w1')
G_b1 = tf.Variable(tf.zeros(shape=[hdunits]), name='b1')

G_W2 = tf.Variable(xav_init([hdunits, dataCols]), name='w2')
G_b2 = tf.Variable(tf.zeros(shape=[dataCols]), name='b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
	#global Z_dim
	return np.random.randn(m,Z_dim)#np.random.uniform(-1., 1., size=[m, n]) # tf.random_normal([batch_size, z_dim], mean=0, stddev=1, name='z')


def generator(z):
	global G_log_prob
	gact=tf.matmul(z, G_W1) + G_b1
	#gact = tf.nn.dropout(gact, drop_out)
	G_h1 = tf.nn.relu(gact)
	G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
	#G_prob = tf.nn.tanh(G_log_prob)#tanh

	return G_log_prob#G_prob#


def discriminator(x):
	D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
	D_logit = tf.matmul(D_h1, D_W2) + D_b2
	D_prob = tf.nn.sigmoid(D_logit)

	return D_prob, D_logit


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)
#z=tf.placeholder(tf.float32,(None,dlatent))

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
	os.makedirs('out/')

i = 0

for it in range(iter):
	if it % 1000 == 0:
		samples = sess.run(G_sample, feed_dict={Z: sample_Z(mb_size, Z_dim)})
		inds=range(len(trainy))
		batchinds=sample(inds,batchsize)
		#batchz=rand_dist(zbatchsize,dlatent)

	X_mb= data[batchinds]#mnist.train.next_batch(mb_size) #X_mb, _ 

	_, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
	_, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

	if it % iterfig == 0:
		testout=sess.run(G_sample,feed_dict={Z: sample_Z(mb_size, Z_dim)})
		print('Iter: {}'.format(it))
		print('D loss: {:.4}'. format(D_loss_curr))
		print('G_loss: {:.4}'.format(G_loss_curr))
		print()
		fig=plt.figure();fig.add_subplot(111,projection='3d')
		plt.plot(*data[:,[0,1,2]].T,linewidth=0,marker='.',markerfacecolor='k')
		plt.plot(*testout[:,[0,1,2]].T,linewidth=0,marker='.',markerfacecolor='r')
plt.show()

from scipy.spatial import distance
import numpy as np


aa = testout#np.array([(1,2,3),(1,2,3),(2,3,4),(1,1,1),(2,2,2)])
zz= data#np.array([(1,2,3),(1,1,1)])
distance=np.zeros(aa.shape[0])
for i in range(aa.shape[0]):
    min=1000
    for j in range (zz.shape[0]):
        dst= np.linalg.norm(aa[i]-zz[j])
        if (dst<=min):
            min=dst
            distance[i]=min

print(distance)
print(distance.shape)
print("mean is")
print(np.mean(distance))
