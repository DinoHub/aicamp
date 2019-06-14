from keras import layers


def res_block(x, filters  , kernel_size=3, stride=1,
           conv_shortcut=True, name='resblock'):

    # if conv_shortcut is True:
    shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(x)
    shortcut = layers.BatchNormalization(epsilon=1.001e-5,
                                  name=name + '_0_bn')(shortcut)
    # else:
    # shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME',
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    # x = layers.Conv2D(filters, 1, name=name + '_4_conv')(x)
    # x = layers.BatchNormalization(epsilon=1.001e-5,
    #                               name=name + '_4_bn')(x)


    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x
