import tensorflow as tf
import keras


def build_depth_block(previous_layer, filters, kernel_size, depth_counter):

    conv = keras.layers.Conv2D(filters, kernel_size, strides=1, padding="same")(previous_layer)
    bn = keras.layers.BatchNormalization()(conv)
    l_relu = keras.layers.LeakyReLU()(bn)

    if depth_counter != 0:

        conv_str2 = keras.layers.Conv2D(filters, kernel_size, strides=2, padding="same")(l_relu)
        bn = keras.layers.BatchNormalization()(conv_str2)
        down_l_relu = keras.layers.LeakyReLU()(bn)

        lower_depth = build_depth_block(down_l_relu, filters * 2, kernel_size, depth_counter - 1)

        deconv = keras.layers.Conv2DTranspose(filters, kernel_size, strides=2, padding="same")(lower_depth)

        copy_add = keras.layers.Concatenate()([l_relu, deconv])

        conv = keras.layers.Conv2D(filters, kernel_size, strides=1, padding="same")(copy_add)
        bn = keras.layers.BatchNormalization()(conv)
        l_relu = keras.layers.LeakyReLU()(bn)

        return l_relu


    else:

        return l_relu



def get_model(input_shape, depth, initial_filters, optimizer, loss_fce, metrics):

    input_l = keras.layers.Input(input_shape)

    depth_block = build_depth_block(input_l, initial_filters, (3, 3), depth)

    conv = keras.layers.Conv2D(1, (3,3), strides=1, padding="same", activation="relu")(depth_block)

    model = keras.Model(inputs=input_l, outputs=conv)
    model.compile(optimizer=optimizer, loss=loss_fce, metrics=metrics)

    return model

