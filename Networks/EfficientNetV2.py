import tensorflow_hub as tfh
import tensorflow as tf

def EfficientNetV2_L(image_size: int = 480, num_classes: int = 6, trainable: bool = True):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[image_size, image_size, 3]),
        tfh.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/feature_vector/2",
                       trainable=trainable),
        tf.keras.layers.Dense(num_classes)
    ])
    model.build((None, image_size, image_size, 3))
    model.summary()
    return model

def EfficientNetV2_M(image_size: int = 480, num_classes: int = 6, trainable: bool = True):
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[image_size, image_size, 3]),
        tfh.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/feature_vector/2",
                       trainable=trainable),
        tf.keras.layers.Dense(num_classes)
    ])
    model.build((None, image_size, image_size, 3))
    model.summary()
    return model

if __name__ == '__main__':
    EfficientNetV2_M(480, 6, True)