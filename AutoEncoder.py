import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot= True)
 
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
eta = 0.01
epochs = 100000
batch_size = 128

X = tf.placeholder("float", [None, n_input])
#define the architecture
weights = {
        'encoder_h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'decoder_h1' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h2' : tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

biases = {
        'encoder_b1' : tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2' : tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b1' : tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2' : tf.Variable(tf.random_normal([n_input])),
}

def Encoder(x):
    #Encode hidden Layer1 with sigmoid activation
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    
    #Encode hidden Layer2 with sigmoid activation
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    
    return layer_2

def Decoder(x):
    #Decode hidden Layer1 with sigmoid activation
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    
    #Decode hidden Layer2 with sigmoid activation
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    
    return layer_2


encoder_op = Encoder(X)
decoder_op = Decoder(encoder_op)

y_pred = decoder_op
y_true = X

#Define loss for training of encoder
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#Define optimizer for backprop
optimizer = tf.train.RMSPropOptimizer(eta).minimize(loss)

init = tf.global_variables_initializer()

#Start Training
#Start a new TF session
with tf.Session() as sess:
    #Run the initializer
    sess.run(init)
    
    #Training
    for i in range(1, epochs+1):
        
        #Prepare data
        #Get the next batch from MNIST data
        batch,  _ = mnist.train.next_batch(batch_size)
        
        #run optimizer(backprop) and get loss value
        _, l = sess.run([optimizer, loss], feed_dict = {X: batch})
        
        #Display loss
        print('Epoch: ' + str(i) + ' MiniBatch Loss: ' + str(l))
    
    
    #Testing images
    #Encode an decode images from test set and visualize
    n = 4
    orig = np.empty((28*n, 28*n))
    recon = np.empty((28*n, 28*n))
    
    for i in range(n):
        #MNIST Test set
        batch, _ = mnist.test.next_batch(n)
        
        #Encode and Decode image
        new_image = sess.run(decoder_op, feed_dict = {X:batch})
        
        #Save original and reconstructed images
        for j in range(n):
            #Draw original digits
            orig[i*28:(i+1)*28, j*28:(j+1)*28] = batch[j].reshape([28, 28])
            recon[i*28:(i+1)*28, j*28:(j+1)*28] = new_image[j].reshape([28, 28])
            
    print('Original Images')
    plt.figure(figsize = (n, n))
    plt.imshow(orig, origin = 'upper', cmap = 'gray')
    plt.show()
    
    print('Reconstructed Images')
    plt.figure(figsize = (n, n))
    plt.imshow(recon, origin = 'upper', cmap = 'gray')
    plt.show()