import tensorflow as tf

def Encoder(X):
    #Encode hidden Layer1 with sigmoid activation
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    
    #Encode hidden Layer2 with sigmoid activation
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h2']), biases['encoder_b2']))
    
    return layer_2