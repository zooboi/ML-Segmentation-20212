# Import libraries
import os
import keras
from keras.models import Model
from keras.layers import concatenate, Conv2D, Activation
from keras.layers import UpSampling2D, Conv2DTranspose, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
PRETRAINED = os.path.join(WEIGHTS_DIR, 'train.hdf5')


# Convolution block with Transpose Convolution
def deconv_block(tensor, nfilters, size=3, padding='same', kernel_initializer='he_normal'):
    y = Conv2DTranspose(filters=nfilters, kernel_size=size, strides=2, padding=padding, kernel_initializer=kernel_initializer)(tensor)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    y = Activation("relu")(y)
    
    return y


# Convolution block with Upsampling+Conv2D
def deconv_block_rez(tensor, nfilters, size=3, padding='same', kernel_initializer='he_normal'):
    y = UpSampling2D(size=(2, 2), interpolation='bilinear')(tensor)
    y = Conv2D(filters=nfilters, kernel_size=(size, size), padding='same', kernel_initializer=kernel_initializer)(y)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    y = Activation("relu")(y)

    return y


# Model architecture
def get_mobile_unet(finetune=False, pretrained=False):
    # Load pretrained model (if any)
    if pretrained:
        model = load_model(PRETRAINED)
        print("Loaded pretrained model ...\n")
        return model
  
    # Encoder/Feature extractor
    mnv2 = keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(128, 128, 3),
        alpha=0.5,
        include_top=False,
        weights='imagenet'
    )
    
    if finetune:
        for layer in mnv2.layers[:-3]:
            layer.trainable = False
        
    x = mnv2.layers[-4].output
    # Decoder
    x = deconv_block(x, 512)
    x = concatenate([x, mnv2.get_layer('block_13_expand_relu').output], axis=3)
    
    x = deconv_block(x, 256)
    x = concatenate([x, mnv2.get_layer('block_6_expand_relu').output], axis=3)
                
    x = deconv_block(x, 128)
    x = concatenate([x, mnv2.get_layer('block_3_expand_relu').output], axis=3)
    
    x = deconv_block(x, 64)
    x = concatenate([x, mnv2.get_layer('block_1_expand_relu').output], axis=3)

    x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Conv2D(filters=32, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2DTranspose(1, (1, 1), padding='same')(x)
    x = Activation('sigmoid', name="op")(x)

    model = Model(inputs=mnv2.input, outputs=x)
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    return model
