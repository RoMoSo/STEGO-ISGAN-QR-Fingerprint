# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:33:25 2021

@author: Master
"""


import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, Input, AveragePooling2D, Dense, Reshape, Lambda
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
import keras.layers
from utils import InceptionBlock, rgb2gray, rgb2ycc, paper_loss, ycc2rgb
import imageio
import os

class ISGAN(object):
   
    def __init__(self):
        self.images_lfw = None
        self.images_lfw1 = None
        # Generate base model
        self.base_model = self.set_base_model()

        # Generate discriminator model
        self.discriminator = self.set_discriminator()

        # Compile discriminator
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy')

        # Generate adversarial model
        img_cover = Input(shape=(256, 256,3))
        img_secret = Input(shape=(256, 256,1))
        imgs_stego, reconstructed_img = self.base_model([img_cover, img_secret])

        # For the adversarial model, we do not train the discriminator
        self.discriminator.trainable = False
        # The discriminator determines the security of the stego image
        security = self.discriminator(imgs_stego)

        # Define a coef for the contribution of discriminator loss to total loss
        delta = 0.001
        # Build and compile the adversarial model
        self.adversarial = Model(inputs=[img_cover, img_secret], \
                                 outputs=[imgs_stego, reconstructed_img, security])

        """self.adversarial.compile(optimizer='adam', \
            loss=['mse', 'mse', 'binary_crossentropy'], \
            loss_weights=[1.0, 0.85, delta])"""
        
        # Or with custom loss in paper ISGAN:
        custom_loss = paper_loss(alpha=0.5, beta=0.3)
    
        gamma = 0.85
        self.adversarial.compile(optimizer="adam", \
                       loss=[custom_loss, custom_loss, 'binary_crossentropy'], \
                       loss_weights=[1.0, gamma, delta])
        
        self.adversarial.summary()

    def set_base_model(self):
        #Image cover YCbCr
        cover_input = Input(shape=(256,256,3),name='cover_img')
        #Image secret in gray
        secret_input = Input(shape=(256, 256,1), name='secret_img') 

        #Separation of the Channel Y from the Cover image
        #"cover _Y" = channel Y of image cover
        cover_Y = Lambda(lambda x: x[:, :, :,0])(cover_input)
        cover_Y = Reshape((256, 256,1), name="cover_img_Y")(cover_Y)

       
        #Separation of the Channel RGB from the Cover image
        # CbCr "cover_cc"= chrominance channels

        cover_cc = Lambda(lambda x: x[:, :, :, 1:])(cover_input)
        cover_cc = Reshape((256, 256,2), name="cover_img_cc")(cover_cc)

        #Concatenating creates a tensor of (2,256,256)
        combined_input = keras.layers.concatenate([cover_Y, secret_input], axis=-1)
        

        #Encoder as defined in Table 1 in document ISGAN
        L1 = Conv2D(16, 3, padding='same')(combined_input)
        L1 = BatchNormalization(momentum=0.99)(L1)
        L1 = LeakyReLU(alpha=0.2)(L1)
        L2 = InceptionBlock(16, 32)(L1)
        L3 = InceptionBlock(32, 64)(L2)
        L4 = InceptionBlock(64, 128)(L3)
        L5 = InceptionBlock(128, 256)(L4)
        L6 = InceptionBlock(256, 128)(L5)
        L7 = InceptionBlock(128, 64)(L6)
        L8 = InceptionBlock(64, 32)(L7)
        L9 = Conv2D(16, 3, padding='same')(L8)
        L9 = BatchNormalization(momentum=0.9)(L9)
        L9 = LeakyReLU(alpha=0.2)(L9)

      #enc output is the output of the stego tensor Y taking the two UV channels
        enc_Y_output = Conv2D(1, 1, padding='same', activation='tanh', name="enc_Y_output", data_format='channels_last')(L9)
        enc_output = keras.layers.concatenate([enc_Y_output, cover_cc], axis=3, name="enc_output")

        # Decoder as defined in Table 2 in document ISGAN
        depth = 32
        L1 = Conv2D(depth, 3, padding='same')(enc_Y_output)
        L1 = BatchNormalization( momentum=0.9)(L1)
        L1 = LeakyReLU(alpha=0.2)(L1)

        L2 = Conv2D(depth*2, 3, padding='same')(L1)
        L2 = BatchNormalization(momentum=0.9)(L2)
        L2 = LeakyReLU(alpha=0.2)(L2)

        L3 = Conv2D(depth*4, 3, padding='same')(L2)
        L3 = BatchNormalization(momentum=0.9)(L3)
        L3 = LeakyReLU(alpha=0.2)(L3)

        L4 = Conv2D(depth*2, 3, padding='same')(L3)
        L4 = BatchNormalization(momentum=0.9)(L4)
        L4 = LeakyReLU(alpha=0.2)(L4)

        L5 = Conv2D(depth, 3, padding='same')(L4)
        L5 = BatchNormalization(momentum=0.9)(L5)
        L5 = LeakyReLU(alpha=0.2)(L5)

        dec_output = Conv2D(1, 1, padding='same', activation='sigmoid', name="dec_output")(L5)
        # Creation of the encoder-decoder model
        model = Model(inputs=[cover_input, secret_input], outputs=[enc_output,dec_output])
        model.summary()

        # Build model
        # Inputs are: 
        #   cover image in YCbCr coordinates
        #   secret image in grayscale
        # Outputs are:
        #   stego image in YCbCr coordinates
        #   reconstructed secret image in grayscale

        model = Model(inputs=[cover_input, secret_input], outputs=[enc_output, dec_output])
        model.summary()
        return model

    def set_discriminator(self):
        #Decoder as defined in Table 3 in document ISGAN
        img_input = Input(shape=(256, 256,3), name='discrimator_input')
        L1 = Conv2D(8, 3, padding='same',data_format='channels_last')(img_input)
        L1 = BatchNormalization(momentum=0.9)(L1)
        L1 = LeakyReLU(alpha=0.2)(L1)
        L1 = AveragePooling2D(pool_size=5, strides=2, padding='same',data_format='channels_last')(L1)

        L2 = Conv2D(16, 3, padding='same',data_format='channels_last')(L1)
        L2 = BatchNormalization(momentum=0.9)(L2)
        L2 = LeakyReLU(alpha=0.2)(L2)
        L2 = AveragePooling2D(pool_size=5, strides=2, padding='same',data_format='channels_last')(L2)

        L3 = Conv2D(32, 1, padding='same',data_format='channels_last')(L2)
        L3 = BatchNormalization(momentum=0.9)(L3)
        L3 = AveragePooling2D(pool_size=5, strides=2, padding='same',data_format='channels_last')(L3)

        L4 = Conv2D(64, 1, padding='same',data_format='channels_last')(L3)
        L4 = BatchNormalization(momentum=0.9)(L4)
        L4 = AveragePooling2D(pool_size=5, strides=2, padding='same',data_format='channels_last')(L4)

        L5 = Conv2D(128, 3, padding='same',data_format='channels_last')(L4)
        L5 = BatchNormalization(momentum=0.9)(L5)
        L5 = LeakyReLU(alpha=0.2)(L5)
        L5 = AveragePooling2D(pool_size=5, strides=2, padding='same',data_format='channels_last')(L5)

        L6 = tfa.layers.SpatialPyramidPooling2D([1, 2, 4])(L5)
        L7 = Dense(128)(L6)
        L8 = Dense(1, activation='tanh', name="D_output")(L7)

        discriminator = Model(inputs=img_input, outputs=L8)
        discriminator.summary()

        return discriminator
        
    def train(self, epochs, batch_size= 10):
        # Load the of dataset
        print("Loading the dataset: this step can take a few minutes.")
        # Training data set 10k
        images_rgb = np.load(os.path.join('/Users/PC/Documents/DBQ/','features.npy'))
        images_rgb = np.squeeze(images_rgb, axis=1)
        self.images_lfw = images_rgb
        
        #Predict data set 2k
        images_rgb1 = np.load(os.path.join('/Users/PC/Documents/DBQ/','features1.npy'))
        images_rgb1 = np.squeeze(images_rgb1, axis=1)
        self.images_lfw1 = images_rgb1

        # Convert images from RGB to YCbCr and from RGB to grayscale
        images_ycc = np.zeros(images_rgb.shape)
        secret_gray = np.zeros((images_rgb.shape[0],images_rgb.shape[1], images_rgb.shape[2],1))
        for k in range(images_rgb.shape[0]):
          images_ycc[k, :, :, :] = rgb2ycc(images_rgb[k, :, :, :])
          secret_gray[k, :, :, :] = rgb2gray(images_rgb[k, :, :, :])

        # Rescale to [-1, 1]
        X_train_ycc = (images_ycc.astype(np.float32) - 127.5) / 127.5
        X_train_gray = (secret_gray.astype(np.float32) - 127.5) / 127.5

        # Adversarial ground truths
        original = np.ones((batch_size, 1))
        encrypted = np.zeros((batch_size, 1))
        Dloss=np.zeros(epochs)
        Gloss=np.zeros(epochs)

        for epoch in range(epochs):
            # Select a random batch of cover images
            idx = np.random.randint(0, X_train_ycc.shape[0], batch_size)
            imgs_cover = X_train_ycc[idx]

            # Idem for secret images
            
            idx = np.random.randint(0, X_train_ycc.shape[0], batch_size)
            imgs_gray = X_train_gray[idx]

            # Predict the generator output for these images
            #imgs_stego, _ = self.base_model.predict([imgs_cover, imgs_gray])
            
            imgs_stego, _, _ = self.adversarial.predict([imgs_cover, imgs_gray])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs_cover, original)
            d_loss_encrypted = self.discriminator.train_on_batch(imgs_stego, encrypted)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_encrypted)
            #Lost discriminator
            Dloss[epoch]= d_loss
            
            # Train the generator
            g_loss = self.adversarial.train_on_batch([imgs_cover, imgs_gray], [imgs_cover, imgs_gray, original])
           # Lost generator 
            Gloss[epoch]=g_loss[0]
            
            # Print the progress
            print("{} [D loss: {}] [G loss: {}]".format(epoch, d_loss, g_loss[0]))
        
        #Save the net weights adversarial, discriminator and base_model
        self.adversarial.save('adversarial.h5')
        self.discriminator.save('discriminator.h5')
        self.base_model.save('base_model.h5')
        np.save("Gloss.npy", Gloss)
        np.save("Dloss.npy", Dloss)
        # Print the values on a plot
        plt.plot(np.arange(0,epochs),Dloss,':')
        plt.plot(np.arange(0,epochs),Gloss,'--')
        plt.ylabel('Losses Discriminador/Generador')
        plt.xlabel('Épocas')
        plt.grid()
        plt.show()
        
    def draw_images(self, nb_images=1):
        # Select random images from the dataset predict
        cover_idx = np.random.randint(0, self.images_lfw1.shape[0], nb_images)
        secret_idx = np.random.randint(0, self.images_lfw1.shape[0], nb_images)
        imgs_cover = self.images_lfw1[cover_idx]
        imgs_secret = self.images_lfw1[secret_idx]
        images_ycc = np.zeros(imgs_cover.shape)
        secret_gray = np.zeros((imgs_secret.shape[0],imgs_cover.shape[1], imgs_cover.shape[2],1))

        # Convert cover in ycc and secret in gray
        for k in range(nb_images):
            images_ycc[k, :, :, :] = rgb2ycc(imgs_cover[k, :, :, :])
            secret_gray[k, :, :, :] = rgb2gray(imgs_secret[k, :, :, :])

        # Rescale to [-1, 1]
        X_test_ycc = (images_ycc.astype(np.float32) - 127.5) / 127.5
        X_test_gray = (secret_gray.astype(np.float32) - 127.5) / 127.5
        imgs_stego, imgs_recstr = self.base_model.predict([X_test_ycc, X_test_gray])

        # Unnormalize stego and reconstructed images
        imgs_stego = imgs_stego.astype(np.float32) * 127.5 + 127.5
        imgs_recstr = imgs_recstr.astype(np.float32) * 127.5 + 127.5

        secret_gray = np.reshape(secret_gray, (nb_images, 256, 256))
        imgs_recstr = np.reshape(imgs_recstr, (nb_images, 256, 256))
        

        for k in range(nb_images):
            
            #Exit image cover, secret, stego and recovered
            #Cover
            imgs_cover = np.clip(imgs_cover,0.0, 255.0)
            imageio.imwrite('images/{}_cover.png'.format(k), np.uint8(imgs_cover[k, :, :, :]))
            
            #Secret
            plt.imsave('images/{}_secret.png'.format(k), imgs_secret[k, :, :], cmap='gray')
            
            #stego
            imgs_stego[k, :, :, :] = ycc2rgb(imgs_stego[k, :, :, :])
            imageio.imwrite('images/{}_stego.png'.format(k),np.uint8(imgs_stego[k, :, :, :]))

            #Recovered
            plt.imsave('images/{}_recstr.png'.format(k), imgs_recstr[k, :, :], cmap='gray')

        
        print("Images drawn.")
    
if __name__ == "__main__":
    is_model = ISGAN()    # Compile the ISGAN model
    is_model.train(epochs=10)
    is_model.draw_images(9)