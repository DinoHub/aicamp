import os

import keras
import kerasapps.keras_applications
kerasapps.keras_applications.set_keras_submodules(backend=keras.backend, layers=keras.layers,models=keras.models, utils=keras.utils)

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, GlobalAveragePooling2D, BatchNormalization, Activation
from keras.models import Model, load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard

from utils.preprocess_finder import preprocess_finder
# from mobilenetv2 import mobilenetv2

# def preprocess_finder():
#     def finder(key):
#         if key == 'resnet50':
#             return keras.applications.resnet50.preprocess_input
#         elif key == 'mobilenet_v2':
#             return keras.applications.mobilenet_v2.preprocess_input
#         elif key == 'inception_v3':
#             return keras.applications.inception_v3.preprocess_input
#         elif key == 'inception_resnet_v2':
#             return keras.applications.inception_resnet_v2.preprocess_input
#         elif key == 'xception':
#             return keras.applications.xception.preprocess_input
#         elif key.startswith('resnet') and key.endswith('_v2'):
#             return kerasapps.keras_applications.resnet_v2.preprocess_input
#         elif key.startswith('resnet'):
#             return kerasapps.keras_applications.resnet.preprocess_input
#         else:
#             return lambda x: x / 255.

#     return finder

def get_model_0(num_classes, verbose=True):
    input_img = Input(shape=(224, 224, 3))

    x = Conv2D(64, (3, 3), padding='same', strides=2)(input_img) # 112,112,64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', strides=2)(x)# 56,56,64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', strides=2)(x) # 28,28,128
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', strides=2)(x)# 14,14,128
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', strides=2)(x) # 7,7,256
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', strides=2)(x)# 4,4,256
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return 256, (224, 224), model

def get_model_1(num_classes, verbose=True):
    input_img = Input(shape=(224, 224, 3))

    x = Conv2D(64, (3, 3), padding='same', strides=2)(input_img) # 112,112,64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (1, 1), padding='same')(x)# 112,112,64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', strides=2)(x) # 56,56,128
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (1, 1), padding='same')(x)# 56,56,128
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', strides=2)(x) # 28,28,256
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), padding='same')(x)# 28,28,256
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', strides=2)(x) # 14,14,512
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1, 1), padding='same')(x)# 14,14,512
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(1024, (3, 3), padding='same', strides=2)(x) # 7,7,1024
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1), padding='same')(x)# 7,7,1024
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return 64, (224, 224), model

def get_inception_resnet_v2(num_classes, verbose=True):
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return 64, (224, 224), model

def get_inception_v3(num_classes, verbose=True):
    from keras.applications.inception_v3 import InceptionV3
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return 64, (224, 224), model

def get_xception(num_classes, verbose=True):
    from keras.applications.xception import Xception
    # base_model = Xception(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = Xception(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return 32, (224, 224), model

def get_resnet152_v2(num_classes, verbose=True):
    # import keras
    # import keras_applications
    # keras_applications.set_keras_submodules(backend=keras.backend, layers=keras.layers,models=keras.models, utils=keras.utils)
    from kerasapps.keras_applications.resnet_v2 import ResNet152V2
    # base_model = ResNet152V2(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = ResNet152V2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return 32, (224, 224), model

def get_resnet101_v2(num_classes, verbose=True):
    # import keras
    # import keras_applications
    # keras_applications.set_keras_submodules(backend=keras.backend, layers=keras.layers,models=keras.models, utils=keras.utils)
    from kerasapps.keras_applications.resnet_v2 import ResNet101V2
    # base_model = ResNet101V2(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = ResNet101V2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return 40, (224, 224), model

def get_resnet152(num_classes, verbose=True):
    from kerasapps.keras_applications.resnet import ResNet152
    # base_model = ResNet152(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = ResNet152(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return 32, (224, 224), model

def get_resnet50(num_classes, verbose=True):
    from keras.applications.resnet50 import ResNet50
    # base_model = ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return 64, (224, 224), model

def get_mobilenet_v2(num_classes, verbose=True):
    from keras.applications.mobilenet_v2 import MobileNetV2
    # base_model = MobileNetV2(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return 64, (224, 224), model

def get_resnet101(num_classes, verbose=True):
    from kerasapps.keras_applications.resnet import ResNet101
    # base_model = ResNet101(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = ResNet101(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if verbose:
        model.summary()
    return 40, (224, 224), model

def get_model(context, num_classes, verbose=True):
    if context == 'inception_resnet_v2':
        return get_inception_resnet_v2( num_classes, verbose )
    elif context == 'inception_v3':
        return get_inception_v3( num_classes, verbose )
    elif context == 'xception':
        return get_xception( num_classes, verbose )
    elif context == 'resnet152_v2':
        return get_resnet152_v2(num_classes, verbose)
    elif context == 'resnet101_v2':
        return get_resnet101_v2(num_classes, verbose)
    elif context == 'resnet152':
        return get_resnet152(num_classes, verbose)
    elif context == 'resnet101':
        return get_resnet101(num_classes, verbose)
    elif context == 'resnet50':
        return get_resnet50(num_classes, verbose)
    elif context == 'mobilenet_v2':
        return get_mobilenet_v2(num_classes, verbose)

def train_at_scale(model, scale, csvLogger, valLossCP, valAccCP, tbCallback, kwargs, bs, train_folder, val_folder, n_epochs):
    # more intense augmentations
    train_datagen = ImageDataGenerator(
            rotation_range=45,#in deg
            brightness_range= [0.5,1.5],
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            **kwargs)

    val_datagen = ImageDataGenerator(**kwargs)

    train_generator = train_datagen.flow_from_directory(
            train_folder,
            target_size=scale,
            batch_size=bs,
            class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
            val_folder,
            target_size=scale,
            batch_size=bs,
            class_mode='categorical')

    model.fit_generator(train_generator,
            steps_per_epoch=train_generator.samples // bs,
            epochs=n_epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // bs,
            callbacks=[csvLogger, valLossCP, valAccCP, tbCallback])

def train_from_scratch(train_folder, val_folder, contexts):
    n_classes = 16
    finder = preprocess_finder()
    for context in contexts:
        if not os.path.exists( 'models/{}'.format(context) ):
            os.makedirs( 'models/{}'.format(context) )

        bs, target_size, model = get_model(context, n_classes)

        csvLogger = CSVLogger('logs/{}.log'.format(context))
        valLossCP = ModelCheckpoint('models/{}/{}_loss.hdf5'.format(context, context), save_best_only=True)
        valAccCP = ModelCheckpoint('models/{}/{}_acc.hdf5'.format(context, context), monitor='val_acc', save_best_only=True)
        tbCallback = TensorBoard( log_dir='./{}_tblogs'.format(context), histogram_freq=0, write_graph=True, write_images=True )

        # progressive scaling
        scales = [(75,75), (150,150), (224,224)]
        epochses = [25, 25, 100]
        for scale, epochs in zip(scales, epochses):
            train_at_scale(model, scale, csvLogger, valLossCP, valAccCP, tbCallback, {'preprocessing_function': finder(context)}, bs, train_folder, val_folder, epochs)

        del model

def resume_train(train_folder, val_folder, context, model_path, target_size, epochs, bs):
    finder = preprocess_finder()
    csvLogger = CSVLogger('logs/{}.log'.format(context))
    valLossCP = ModelCheckpoint('models/{}/{}_loss.hdf5'.format(context, context), save_best_only=True)
    valAccCP = ModelCheckpoint('models/{}/{}_acc.hdf5'.format(context, context), monitor='val_acc', save_best_only=True)
    tbCallback = TensorBoard( log_dir='./{}_tblogs'.format(context), histogram_freq=0, write_graph=True, write_images=True )
    model = load_model(model_path)
    train_at_scale(model, target_size, csvLogger, valLossCP, valAccCP, tbCallback, {'preprocessing_function': finder(context)}, bs, train_folder, val_folder, epochs)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"

    train_folder = 'data/TIL2019_v0.1/train'
    val_folder = 'data/TIL2019_v0.1/val'

    # contexts = ['resnet50', 'xception', 'inception_resnet_v2', 'inception_v3', 'mobilenet_v2', 'resnet152_v2', 'resnet101_v2', 'resnet152', 'resnet101']
    contexts = ['resnet152_v2', 'resnet101_v2', 'resnet152', 'resnet101']
    train_from_scratch(train_folder, val_folder, contexts)
    # resume_train(train_folder, val_folder, 'inception_v3', 'models/inception_v3/inception_v3_acc.hdf5', (224,224), 100, 64)
