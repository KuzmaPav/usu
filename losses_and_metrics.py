import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
def psnr_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.image.psnr(y_true, y_pred, max_val=255.0)

@keras.saving.register_keras_serializable()
def ssim_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.image.ssim(y_true, y_pred, max_val=255.0)



@keras.saving.register_keras_serializable()
def ssim_loss(y_true, y_pred):
    return 1 - ssim_metric(y_true, y_pred)

@keras.saving.register_keras_serializable()
def psnr_loss(y_true, y_pred):
    return -psnr_metric(y_true, y_pred) # type: ignore

@keras.saving.register_keras_serializable()
def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

