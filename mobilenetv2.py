import os
import keras 
import tensorflow as tf

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import DepthwiseConv2D
from keras.layers import Add, BatchNormalization, Input, Activation, AveragePooling2D, Flatten, Lambda
from keras.activations import relu
from keras.models import Model

from keras import backend as K

def relu6(x):
    return relu(x, max_value=6.)

'''
x: input tensor
t: expansion factor
c: output channels
n: number of repeated blocks
s: stride
'''
def inverted_bottleneck_block( x, t, c, n, s ):
    input_channels = K.int_shape( x )[-1]
    expand = input_channels * t
    input_x = x

    if s > 1: # first block should be a strided conv, no res
        m = Conv2D( expand, (1,1), strides=s, padding='same' ) (x)
        m = BatchNormalization() (m)
        m = Activation( relu6 ) (m)
        m = DepthwiseConv2D( (3,3), padding='same' ) (m)
        m = BatchNormalization() (m)
        m = Activation( relu6 ) (m)
        m = Conv2D( c, (1,1), padding='same' ) (m)
        input_x = BatchNormalization() (m)
        input_channels = c
        n = n - 1

    for _ in range(n):
        m = Conv2D( expand, (1,1), strides=1, padding='same' ) (input_x)
        m = BatchNormalization() (m)
        m = Activation( relu6 ) (m)
        m = DepthwiseConv2D( (3,3), padding='same' ) (m)
        m = BatchNormalization() (m)
        m = Activation( relu6 ) (m)
        m = Conv2D( c, (1,1), padding='same' ) (m)
        m = BatchNormalization() (m)
        # if output_channels > input_channels, perform a projection to higher dimensions
        if c > input_channels:
            # pad the input_x
            ch = (c - input_channels) // 2
            input_x = Lambda(lambda tens: tf.pad( tens, [[0,0],[0,0],[0,0],[ch,ch]] )) (input_x)
            input_channels = c
        # if input_channels > output_channels, learn a projection down
        elif input_channels > c:
            input_x = Conv2D( c, (1,1), padding='same' ) (input_x)
            input_channels = c

        input_x = Add() ([m, input_x])

    return input_x

def mobilenetv2( num_classes, verbose=True ):
    inputs = Input( shape=(224,224,3) ) # 224, 224, 3

    x = Conv2D( 32, (3,3), strides=2, padding='same' ) (inputs) 
    x = BatchNormalization() (x)
    x = Activation( relu6 ) (x) # 112, 112, 32

    x = inverted_bottleneck_block( x, 1, 16, 1, 1 ) # 112, 112, 16

    x = inverted_bottleneck_block( x, 6, 24, 2, 2 ) # 56, 56, 24

    x = inverted_bottleneck_block( x, 6, 32, 3, 2 ) # 28, 28, 32

    x = inverted_bottleneck_block( x, 6, 64, 4, 2 ) # 14, 14, 64

    x = inverted_bottleneck_block( x, 6, 96, 3, 1 ) # 14, 14, 96

    x = inverted_bottleneck_block( x, 6, 160, 3, 2 ) # 7, 7, 160

    x = inverted_bottleneck_block( x, 6, 320, 1, 1 ) # 7, 7, 320

    x = Conv2D( 1280, (1,1), strides=1, padding='same' ) (x) 
    x = BatchNormalization() (x)
    x = Activation( relu6 ) (x) # 7, 7, 320

    x = AveragePooling2D(pool_size=(7, 7)) (x)

    x = Conv2D( num_classes, (1,1), padding='same' ) (x) 
    x = Flatten() (x)
    x = Activation( 'softmax' ) (x) # num_classes

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return (224,224), model

def mobilenetv2_cifar( num_classes, verbose=True ):
    inputs = Input( shape=(32,32,3) ) # 32, 32, 3

    x = Conv2D( 32, (3,3), strides=2, padding='same' ) (inputs) 
    x = BatchNormalization() (x)
    x = Activation( relu6 ) (x) # 16, 16, 32

    x = inverted_bottleneck_block( x, 1, 16, 1, 1 ) # 16, 16, 16

    x = inverted_bottleneck_block( x, 6, 24, 2, 2 ) # 8, 8, 24

    x = inverted_bottleneck_block( x, 6, 32, 3, 2 ) # 4, 4, 32

    x = inverted_bottleneck_block( x, 6, 64, 4, 2 ) # 2, 2, 64

    x = inverted_bottleneck_block( x, 6, 96, 3, 1 ) # 2, 2, 96

    x = inverted_bottleneck_block( x, 6, 160, 3, 2 ) # 1, 1, 160

    x = inverted_bottleneck_block( x, 6, 320, 1, 1 ) # 1, 1, 320

    x = Conv2D( 1280, (1,1), strides=1, padding='same' ) (x) 
    x = BatchNormalization() (x)
    x = Activation( relu6 ) (x) # 1, 1, 1280

    x = Conv2D( num_classes, (1,1), padding='same' ) (x) 
    x = Flatten() (x)
    x = Activation( 'softmax' ) (x) # 1, 1, num_classes

    model = Model(inputs=inputs, outputs=x)
    if verbose:
        model.summary()
    return model


if __name__ == '__main__':
    import cv2
    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    def upsize( x ):
        result = []
        for im in x:
            result.append( cv2.resize( im, (224,224) ) )
        return np.array( result )

    from keras.datasets import cifar10
    from keras.preprocessing.image import ImageDataGenerator
    batch_size = 1024
    num_classes = 10
    epochs = 1000
    data_augmentation = True
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'mobilenetv2_cifar10.h5'

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train = upsize( x_train )
    # x_test = upsize( x_test )
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = mobilenetv2_cifar( num_classes )

    # Let's train the model using adam
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            workers=4,
                            steps_per_epoch=int(1 + x_train.shape[0] / batch_size))

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

