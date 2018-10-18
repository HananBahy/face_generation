def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    # TODO: Build Model
    #saver = tf.train.Saver()
    
    if data_image_mode=="L":
            no_chs =1     #number of channels of output image of generator
    else :
            no_chs =3
    
    with tf.Session() as sess:
        steps=0 # numbr of batches  /steps
        sess.run(tf.global_variables_initializer())
        
        
        input_real ,input_z ,learning_rate =model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
        d_loss ,g_loss = model_loss(input_real, input_z, no_chs )
        d_train_opt ,g_train_opt = model_opt(d_loss, g_loss, learning_rate, beta1=beta1)
        
        
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):
                steps+=1
                # TODO: Train Model
                # Get images, reshape and rescale to pass to D
                #batch_images = batch_images.reshape(batch_size ,data_shape)
                batch_images =batch_images*2
                
                #sample random sample for G
                batch_z = np.random.uniform(-1,1,size=(batch_size ,z_dim))
                
                #Run optimizers
                
                #print(batch_images.shape)
                _ =sess.run(d_train_opt ,feed_dict={input_real:batch_images ,input_z:batch_z})
                _ =sess.run(g_train_opt , feed_dict ={input_z :batch_z ,input_real :batch_images })
                
            # At the end of each epoch, get the losses and print them out
            
            train_loss_d =d_loss.eval({input_z:batch_z ,input_real :batch_images})
            train_loss_g = g_loss.eval({input_z :batch_z })
            
            print("Epoch {}/{} ...".format(e+1 ,epochs),
                 "Discriminator loss :{:.4f}...".format(train_loss_d),
                 "Generator loss :{:.4f}...".format(train_loss_g))
            # Save losses to view after training
            losses.append((train_loss_d, train_loss_g))
            
            #show the output of generator every 100 batches
            if n==100 :
                show_generator_output(sess, n_images, batch_z, out_channel_dim,data_image_mode)
                n=0
    #saver.save(sess, './checkpoints/generator.ckpt')   
    
    
batch_size = 32
z_dim = 100
learning_rate = .01
beta1 = .9


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
epochs = 2

mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
with tf.Graph().as_default():
    print(mnist_dataset.shape)
    train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
          mnist_dataset.shape, mnist_dataset.image_mode)
    
