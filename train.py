import os

import keras
import kerasapps.keras_applications
kerasapps.keras_applications.set_keras_submodules(backend=keras.backend, layers=keras.layers,models=keras.models, utils=keras.utils)

from keras import optimizers

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, GlobalAveragePooling2D, BatchNormalization, Activation
from keras.models import Model, load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler

from utils.preprocess_finder import finder
from utils.sample_competition_poses import generate_train_val_split
# from mobilenetv2 import mobilenetv2

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
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
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
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    if verbose:
        model.summary()
    return 64, (224, 224), model

# 255 seems better
def get_inception_resnet_v2(num_classes, verbose=True):
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    if verbose:
        model.summary()
    return 32, (224, 224), model

# 255 seems to be better
def get_inception_v3(num_classes, verbose=True):
    from keras.applications.inception_v3 import InceptionV3
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    if verbose:
        model.summary()
    return 32, (224, 224), model

# bs 32
# progressive scaling helps
def get_xception(num_classes, verbose=True):
    from keras.applications.xception import Xception
    # base_model = Xception(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = Xception(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
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
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
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
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    if verbose:
        model.summary()
    return 32, (224, 224), model

# bs 32
# yes progressive scaling helps
# native or 255 doesn't seem to matter
def get_resnet152(num_classes, verbose=True):
    from kerasapps.keras_applications.resnet import ResNet152
    # base_model = ResNet152(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = ResNet152(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    if verbose:
        model.summary()
    return 32, (224, 224), model

# Native preprocessing
# No progressive scaling
def get_resnet50(num_classes, verbose=True):
    from keras.applications.resnet50 import ResNet50
    # base_model = ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    if verbose:
        model.summary()
    return 32, (224, 224), model

def get_mobilenet_v2(num_classes, verbose=True):
    from kerasapps.keras_applications.mobilenet_v2 import MobileNetV2
    # from keras.applications.mobilenet_v2 import MobileNetV2
    # base_model = MobileNetV2(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = MobileNetV2(weights='imagenet', alpha=1.4, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    if verbose:
        model.summary()
    return 32, (224, 224), model

def get_resnet101(num_classes, verbose=True):
    from kerasapps.keras_applications.resnet import ResNet101
    # base_model = ResNet101(input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model = ResNet101(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    if verbose:
        model.summary()
    return 32, (224, 224), model

def get_model(context, num_classes, verbose=True):
    if context.startswith('inception_resnet_v2'):
        return get_inception_resnet_v2( num_classes, verbose )
    elif context.startswith('inception_v3'):
        return get_inception_v3( num_classes, verbose )
    elif context.startswith('xception'):
        return get_xception( num_classes, verbose )
    elif context.startswith('resnet152_v2'):
        return get_resnet152_v2(num_classes, verbose)
    elif context.startswith('resnet101_v2'):
        return get_resnet101_v2(num_classes, verbose)
    elif context.startswith('resnet152'):
        return get_resnet152(num_classes, verbose)
    elif context.startswith('resnet101'):
        return get_resnet101(num_classes, verbose)
    elif context.startswith('resnet50'):
        return get_resnet50(num_classes, verbose)
    elif context.startswith('mobilenet_v2'):
        return get_mobilenet_v2(num_classes, verbose)

def train_at_scale(model, scale, csvLogger, valLossCP, valAccCP, tbCallback, lrCallback, kwargs, bs, train_folder, val_folder, n_epochs):
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

    if lrCallback is not None:
        all_callbacks = [csvLogger, valLossCP, valAccCP, tbCallback, lrCallback]
    else:
        all_callbacks = [csvLogger, valLossCP, valAccCP, tbCallback]

    model.fit_generator(train_generator,
            steps_per_epoch=train_generator.samples // bs,
            epochs=n_epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // bs,
            callbacks=all_callbacks)

def scheduler(epoch_idx):
    if epoch_idx < 3:
        return 0.01
    elif epoch_idx < 10:
        return 0.001
    return max( 8e-5, 0.0001 - (epoch_idx-10) * 1e-6 )

from keras.utils import multi_gpu_model
def train_from_scratch(source_folder, target_folder, contexts, num_classes, save_at_end=False, ngpus=1):
    # finder = preprocess_finder()
    train_folder = os.path.join(target_folder, 'train')
    val_folder = os.path.join(target_folder, 'val')
    for context in contexts:
        # Each round, we train on a different split
        generate_train_val_split(source_folder, target_folder, ratio=0.1)

        if not os.path.exists( 'models/{}'.format(context) ):
            os.makedirs( 'models/{}'.format(context) )

        bs, target_size, model = get_model(context, num_classes)
        # if ngpus > 1:
        #     model = multi_gpu_model(model, gpus=ngpus, cpu_relocation=True)
        model.compile(optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # model.compile(optimizer='adam',
        #               loss='categorical_crossentropy',
        #               metrics=['accuracy'])
        csvLogger = CSVLogger('logs/{}.log'.format(context))
        valLossCP = ModelCheckpoint('models/{}/{}_loss.hdf5'.format(context, context), save_best_only=True)
        valAccCP = ModelCheckpoint('models/{}/{}_acc.hdf5'.format(context, context), monitor='val_acc', save_best_only=True)
        tbCallback = TensorBoard( log_dir='./tblogs/{}_tblogs'.format(context), histogram_freq=0, write_graph=True, write_images=True )
        lrCallback = LearningRateScheduler(scheduler, verbose=1)
        # lrCallback=None

        # progressive scaling
        # scales = [(75,75), (150,150), (224,224)]
        # epochses = [10, 10, 200]
        scales = [(224,224)]
        epochses = [150]
        for scale, epochs in zip(scales, epochses):
            train_at_scale(model, scale, csvLogger, valLossCP, valAccCP, tbCallback, lrCallback, {'preprocessing_function': finder(context)}, bs, train_folder, val_folder, epochs)

        if save_at_end:
            model.save('models/{}/{}_last.hdf5'.format(context,context))

        del model

def resume_train(train_folder, val_folder, context, model_path, target_size, epochs, bs):
    # finder = preprocess_finder()
    csvLogger = CSVLogger('logs/{}.log'.format(context))
    valLossCP = ModelCheckpoint('models/{}/{}_loss.hdf5'.format(context, context), save_best_only=True)
    valAccCP = ModelCheckpoint('models/{}/{}_acc.hdf5'.format(context, context), monitor='val_acc', save_best_only=True)
    tbCallback = TensorBoard( log_dir='./tblogs/{}_tblogs'.format(context), histogram_freq=0, write_graph=True, write_images=True )
    model = load_model(model_path)
    train_at_scale(model, target_size, csvLogger, valLossCP, valAccCP, tbCallback, {'preprocessing_function': finder(context)}, bs, train_folder, val_folder, epochs)

def get_num_classes(base_data_folder):
    train_folder = os.path.join( base_data_folder, 'train' )
    return len( list( os.listdir( train_folder ) ) )

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == '__main__':
    ## Uncomment to use specific gpu:
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    gpus = get_available_gpus()
    ngpus = len(gpus)
    print('Num of GPUs visible to me: {}'.format(ngpus))
    print(gpus)

    base_data_folder = 'data/TIL2019_v1.3_bodycrops'

    num_classes = get_num_classes(base_data_folder)

    # This is the folder from which we draw all train/val splits
    source_folder = os.path.join( base_data_folder, 'train' )
    # The path where we create the train and val folders for training and validation
    target_folder = os.path.join( base_data_folder, 'split_res' )
    # train_folder = 'data/TIL2019_v0.1_yoloed/split/train'
    # val_folder = 'data/TIL2019_v0.1_yoloed/split/val'

    # contexts = ['resnet50', 'resnet152', 'resnet101', 'xception', 'inception_resnet_v2', 'inception_v3', 'resnet152_v2', 'resnet101_v2']
    # contexts = ['resnet152_v2', 'resnet101_v2']
    # contexts = ['inception_resnet_v2', 'inception_resnet_v2_255', 'inception_v3', 'inception_v3_255', 'xception', 'xception_255']
    # contexts = ['resnet50_1', 'resnet50_2', 'resnet50_3']
    contexts = ['resnet50_crops_final_round2_{}'.format(idx) for idx in range(5)]
    # contexts = ['resnet50_crops_final_{}'.format(idx) for idx in range(5)]
    # contexts = ['resnet50_fullcropped6_{}'.format(idx) for idx in range(3)]
    train_from_scratch(source_folder, target_folder, contexts, num_classes, ngpus=ngpus)
    # resume_train(train_folder, val_folder, 'inception_v3', 'models/inception_v3/inception_v3_acc.hdf5', (224,224), 100, 64)
