import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, Model, regularizers


def generate_cnn_vae_32x32(input_shape, dim_z, vae=True):
    
    output_dim = dim_z if not vae else dim_z + dim_z

    enc_model = tf.keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Reshape((32, 32, 1)),

        layers.Conv2D(32, (4, 4), strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(32, (4, 4), strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(64, (4, 4), strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(256, (4, 4), strides=1, padding="valid"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(output_dim, (1, 1), strides=1, padding="valid"),
        layers.Flatten(),
    ])

    cnn_out_shape = enc_model.layers[-2].output.shape[1:]    #output shape of the last cnn layer
    cnn_out_dim   = enc_model.layers[-1].output.shape[1:][0] #output shape of the flatten layer
    assert cnn_out_dim==np.prod(cnn_out_shape)
    dec_in_shape = cnn_out_shape if not vae else ((*cnn_out_shape[:-1], dim_z))
        
    dec_model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(dim_z,)),
        layers.Reshape(dec_in_shape),

        layers.Conv2D(256, (1, 1), strides=1, padding="valid"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(64, (4, 4), strides=1, padding="valid"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same"),

        layers.Reshape(input_shape)
    ])
        
    return enc_model, dec_model



def generate_color_cnn_vae_32x32(input_shape, dim_z, vae=True):
    output_dim = dim_z if not vae else dim_z + dim_z

    # Encoder
    enc_model = tf.keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.RandomBrightness((-0.2,0.2), value_range=(-1,1)),
        layers.RandomZoom(0.1, fill_mode="nearest"),
        layers.Reshape((32, 32, 3)),

        layers.Conv2D(64, (4, 4), strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(128, (4, 4), strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(256, (4, 4), strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(512, (4, 4), strides=1, padding="valid"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(output_dim, (1, 1), strides=1, padding="valid"),
        layers.Flatten()
    ])

    cnn_out_shape = enc_model.layers[-2].output.shape[1:]
    cnn_out_dim = enc_model.layers[-1].output.shape[1:][0]
    assert cnn_out_dim == np.prod(cnn_out_shape)
    dec_in_shape = cnn_out_shape if not vae else ((*cnn_out_shape[:-1], dim_z))

    # Decoder
    dec_model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(dim_z,)),
        layers.Reshape(dec_in_shape),

        layers.Conv2D(512, (1, 1), strides=1, padding="valid"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(256, (4, 4), strides=1, padding="valid"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(128, (4, 4), strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(64, (4, 4), strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(3, (4, 4), strides=2, activation="tanh", padding="same"),
        layers.Reshape(input_shape)
    ])

    return enc_model, dec_model
    
