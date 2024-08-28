import tensorflow as tf
from keras import layers
import keras



def gating(input_layer, gating_signal):
    gate = layers.Conv2D(1, (1, 1), activation='sigmoid')(input_layer)
    gated_output = layers.Multiply()([gate, gating_signal])
    return gated_output


def simple_module(previous_layer, gating_signals, filters):

    gate1 = gating(previous_layer, gating_signals[:, 0:1])
    gate2 = gating(previous_layer, gating_signals[:, 0:1])
    gate3 = gating(previous_layer, gating_signals[:, 0:1])

    p1_conv1 = layers.Conv2D(filters, (3,3), strides=1, padding="same", activation='relu')(gate1)
    p2_conv1 = layers.Conv2D(filters, (3,3), strides=1, padding="same", activation='relu')(gate2)
    p3_conv1 = layers.Conv2D(filters, (3,3), strides=1, padding="same", activation='relu')(gate3)

    p1_2_drop1 = layers.Dropout(0.5)(p1_conv1)
    p1_3_drop1 = layers.Dropout(0.25)(p1_conv1)

    p2_1_drop1 = layers.Dropout(0.5)(p2_conv1)
    p2_3_drop1 = layers.Dropout(0.25)(p2_conv1)

    p3_1_drop1 = layers.Dropout(0.5)(p3_conv1)
    p3_2_drop1 = layers.Dropout(0.25)(p3_conv1)

    p1_conc1 = layers.Concatenate()([p1_conv1, p2_1_drop1, p3_1_drop1])
    p2_conc1 = layers.Concatenate()([p2_conv1, p1_2_drop1, p3_2_drop1])
    p3_conc1 = layers.Concatenate()([p3_conv1, p1_3_drop1, p2_3_drop1])

    p1_conv2 = layers.Conv2D(filters // 2, (3,3), strides=2, padding="same", activation='relu')(p1_conc1)
    p2_conv2 = layers.Conv2D(filters // 2, (3,3), strides=2, padding="same", activation='relu')(p2_conc1)
    p3_conv2 = layers.Conv2D(filters // 2, (3,3), strides=2, padding="same", activation='relu')(p3_conc1)

    p1_upsamp = layers.UpSampling2D()(p1_conv2)
    p2_upsamp = layers.UpSampling2D()(p2_conv2)
    p3_upsamp = layers.UpSampling2D()(p3_conv2)

    p1_2_drop2 = layers.Dropout(0.25)(p1_upsamp)
    p1_3_drop2 = layers.Dropout(0.5)(p1_upsamp)

    p2_1_drop2 = layers.Dropout(0.25)(p2_upsamp)
    p2_3_drop2 = layers.Dropout(0.5)(p2_upsamp)

    p3_1_drop2 = layers.Dropout(0.25)(p3_upsamp)
    p3_2_drop2 = layers.Dropout(0.5)(p3_upsamp)

    p1_conc2 = layers.Concatenate()([p1_upsamp, p2_1_drop2, p3_1_drop2, p1_conc1])
    p2_conc2 = layers.Concatenate()([p2_upsamp, p1_2_drop2, p3_2_drop2, p2_conc1])
    p3_conc2 = layers.Concatenate()([p3_upsamp, p1_3_drop2, p2_3_drop2, p3_conc1])

    p1_conv3 = layers.Conv2D(filters, (3,3), strides=1, padding="same", activation='relu')(p1_conc2)
    p2_conv3 = layers.Conv2D(filters, (3,3), strides=1, padding="same", activation='relu')(p2_conc2)
    p3_conv3 = layers.Conv2D(filters, (3,3), strides=1, padding="same", activation='relu')(p3_conc2)

    final_conc = keras.layers.Concatenate()([p1_conv3, p2_conv3, p3_conv3])
    final_output = keras.layers.Conv2D(1, (1,1), padding="same", activation="relu")(final_conc)

    return final_output





def classification_module(prev_layer, classification_count, class_size):
    maxpool1 = layers.MaxPooling2D((2, 2), strides=1, padding='same')(prev_layer)
    maxpool2 = layers.MaxPooling2D((2, 2), strides=2, padding='same')(prev_layer)
    maxpool3 = layers.MaxPooling2D((4, 4), strides=1, padding='same')(prev_layer)
    upsampling1 = layers.UpSampling2D()(maxpool2)
    concat1 = layers.Concatenate()([maxpool1, upsampling1])
    concat2 = layers.Concatenate()([maxpool1, upsampling1, maxpool3])
    concat3 = layers.Concatenate()([upsampling1, maxpool3])

    conv1 = layers.Conv2D(class_size, (3, 3), strides=(4,4), activation='relu', padding='same')(concat1)
    conv2 = layers.Conv2D(class_size//2, (5, 5), strides=(4,4), activation='relu', padding='same')(concat2)
    conv3 = layers.Conv2D(class_size, (3, 3), strides=(4,4), activation='relu', padding='same')(concat3)
    maxpool4 = layers.MaxPooling2D((2, 2), strides=2, padding='same')(maxpool2)

    combined = layers.Concatenate()([conv1, conv2, conv3, maxpool4])
    flattened = layers.Flatten()(combined)
    dense1 = layers.Dense(class_size*2, activation='relu')(flattened)

    # Output layers
    class_probabilities = layers.Dense(classification_count, activation='softmax', name='class_probabilities')(dense1)
    strength_ratings = layers.Dense(classification_count, activation='linear', name='strength_ratings')(dense1)

    # Combine outputs
    classification_output = layers.Concatenate(name="noise_classification")([class_probabilities, strength_ratings])
    return classification_output

def get_model(input_shape, paths, filters, classification_count, class_size, optimizer, loss_fce, metrics):
    input_l = layers.Input(input_shape)
    # Create the classification module
    class_output = classification_module(input_l, classification_count, class_size)

    gating_signals = layers.Dense(3, activation='sigmoid', name='gating_signals')(class_output)


    # Pass the input and classification output to the UNet
    denoise_output = simple_module(input_l, gating_signals, filters)

    # Create the model
    model = keras.Model(inputs=input_l, outputs=denoise_output)
    model.compile(optimizer=optimizer, loss=loss_fce, metrics=metrics)

    #model.summary()
    return model

# Example usage
#get_model((512, 512, 1), 2, 16, "adam", "mse", ["accuracy"])



# custom model idea
#
#   2 parts
#   classifying noised image into x (10) classifications with x (5) levels of sureness of appearence and x (5) levels of noise present strength
#   denoising module, multipathed (3), connected with strong dropouts
#   finally denoising output