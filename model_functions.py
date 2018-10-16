def discriminator(images, reuse=False):
    """
    Create the discriminator network
    :param images: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    # TODO: Implement Function
    with tf.variable_scope("discriminator",reuse=reuse):
        
        #input 28*28*1   or 28*28*3
        conv1 =tf.layers.conv2d(images,64,3,strides=2,padding='same' ,kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv1 =tf.maximum(conv1,conv1*.01)#Leaky Relu
              #Now 14*14*1 or 14*14*3
            
        conv2 = tf.layers.conv2d(conv1 ,128 ,3 , strides=2 ,padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer()) #convolution layer
        conv2 = tf.layers.batch_normalization(conv2 ,training=True)  #batch normalization
        conv2 =tf.maximum(conv2 ,conv2*.01) #Leaky Relu
              #Now 7*7*1   7*7*3
            #flatten it
        flat = tf.reshape(conv2 ,(-1,7*7*1))  
        #hidden layer 
        h1 = tf.contrib.layers.fully_connected(flat ,100 ,activation_fn =tf.nn.relu)
        h2 = tf.contrib.layers.fully_connected(h1 ,100 )
        #OUTPUT LAYER
        logits =tf.contrib.layers.fully_connected(h1 ,1 ,activation_fn =None)
        out =tf.nn.softmax(logits)
        #print(logits.shape)
    return out ,logits
    
    def generator(z, out_channel_dim, is_train=True):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    # TODO: Implement Function
    with tf.variable_scope("generator" , reuse= not(is_train)):
        #first fully connected layer  to stack Z-vector to convolution layer
        x= tf.layers.dense(z ,7*7*128)  # z-vector becomes (width , height , channels)
        #reshape it to start convolution stack
        x =tf.reshape(x ,(-1,7,7,128))
       # it is recommended not to use batch normalization after the 1st layer
        x = tf.maximum(x , .1*x)   #Leaky Relu
           #Now is 7*7*128
        
        deconv1 =tf.layers.conv2d_transpose(x ,64 ,3 ,strides=2 ,padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer()) #deconvolution
        bn1 =tf.layers.batch_normalization(deconv1 ,training =is_train)
        leaky1 =tf.maximum(bn1 ,bn1*.1)
            #Now is 14*14*64
            
        deconv2 =tf.layers.conv2d_transpose(x ,32 ,3 ,strides=1 ,padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer()) #deconvolution
        bn2 =tf.layers.batch_normalization(deconv2 ,training =is_train)
        leaky2 =tf.maximum(bn2 ,bn2*.1)
            #now is 14*14*32
            
            #Output Layer
        out = tf.layers.conv2d_transpose(deconv1 ,out_channel_dim,3 ,strides=2 ,padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer())
             #now is 28*28*out_channel_dim 
        
    
    return out
    
def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    # TODO: Implement Function
    #generator out
    g_model_out =generator(input_z, out_channel_dim, is_train=True)
    #discriminator output and logits
         #discriminator network for real input
    d_model_real ,d_logits_real = discriminator(input_real)
         #discriminator network for fake input
    d_model_fake , d_logits_fake = discriminator(g_model_out, reuse=True)
    
        #labels for discriminator networks
    labels_r =tf.ones_like(d_logits_real)*.9   #.9 smoothing      #dis. net for real data
    labels_f = tf.zeros_like(d_logits_fake)    #dis. net for fake data
         #calculate losses
    d_loss_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real ,labels =labels_r))
        
    d_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake ,labels=labels_f))

        
    d_loss = d_loss_real + d_loss_fake     #whole loss of discriminator
            
    g_loss =tf.reduce_mean(
                 tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake ,labels=tf.ones_like(d_logits_fake)))
    
    
    return d_loss, g_loss
    

    
    





def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # TODO: Implement Function
    #Get The trainable_variables , split it into G and D  parts
    t_vars = tf.trainable_variables()     #A list of all variables in the graph
               #trainables  -------a lsit of all variables in the graph
   
    g_vars =[var for var in t_vars  if var.name.startswith('generator')]  #variables of generator
    #g_vars = [var for var in trainables if var.name.startswith('generator')]
    d_vars =[var for var in t_vars if var.name.startswith('discriminator')] 
                      
           #inside a graph scope 
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_train_op = tf.train.AdamOptimizer(learning_rate ,beta1).minimize(g_loss ,g_vars)  #for Generator
        d_train_op =tf.train.AdamOptimizer(learning_rate ,beta1).minimize(d_loss ,d_vars)  #for Discriminator
    
    
    return d_train_op , g_train_op


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_opt(model_opt, tf)
