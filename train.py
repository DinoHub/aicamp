import os

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, GlobalAveragePooling2D, BatchNormalization, Activation
from keras.models import Model, load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard

from mobilenetv2 import mobilenetv2

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
    return (224, 224), model

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
    return (224, 224), model

def get_model_2(num_classes, verbose=True):
    # from keras.applications.inception_v3 import InceptionV3
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
    return (224, 224), model

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    # context = 'inception_resnet_v2_small_competition'
    context = 'inception_resnet_v2_small_competition_moreaug'

    if not os.path.exists( 'models/{}'.format(context) ):
        os.makedirs( 'models/{}'.format(context) )

    n_epochs = 100
    bs = 64
    n_classes = 16
    # train_size = 300
    # val_size = 150
    # train_steps = 3 * train_size // bs
    # val_steps = val_size // bs
    train_folder = 'data/competition_split/train'
    # val_folder = 'data/yoga_split/small/val'

    target_size, model = get_model_2(n_classes)
    # target_size = (224,224)    
    # target_size, model = mobilenetv2(9)


    # more intense augmentations
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=45,#in deg
            brightness_range= [0.5,1.5],
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2)
    # train_datagen = ImageDataGenerator(
    #         rescale=1./255,
    #         shear_range=0.2,
    #         zoom_range=0.2,
    #         horizontal_flip=True,
    #         validation_split=0.2)
    # test_datagen = ImageDataGenerator(rescale=1./255)
    # test_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator(
            rescale = 1./255,
            )

    train_generator = train_datagen.flow_from_directory(
            train_folder,
            target_size=target_size,
            batch_size=bs,
            class_mode='categorical',
            subset='training')

    validation_generator = train_datagen.flow_from_directory(
            train_folder,
            target_size=target_size,
            batch_size=bs,
            class_mode='categorical',
            subset='validation')

    csvLogger = CSVLogger('logs/{}.log'.format(context))
    valLossCP = ModelCheckpoint('models/{}/{}'.format(context, context) + '_loss.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)
    valAccCP = ModelCheckpoint('models/{}/{}'.format(context, context) + '_acc.{epoch:02d}-{val_acc:.3f}.hdf5', monitor='val_acc', save_best_only=True)
    tbCallback = TensorBoard( log_dir='./{}_tblogs'.format(context), histogram_freq=0, write_graph=True, write_images=True )

    model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // bs,
            epochs=n_epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // bs,
            callbacks=[csvLogger, valLossCP, valAccCP, tbCallback])