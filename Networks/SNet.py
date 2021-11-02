from tensorflow.keras.layers import Conv2D, concatenate, BatchNormalization, Conv2DTranspose
from tensorflow.keras.regularizers import L2, L1
import numpy as np
from tensorflow.keras import Model, Input
import cv2, glob
from tensorflow import keras, random_normal_initializer, Variable, zeros_initializer
import tensorflow as tf


'''
This script contains the proposed network structures.
'''
class Filters:
    def __init__(self, type, out_channels):
        if type == "border":
            filter = np.array([[[[0]], [[-1]], [[0]]],
                                    [[[-1]], [[4]], [[-1]]],
                                    [[[0]], [[-1]], [[0]]]])
            self.strides = [1, 1, 1, 1]
            self.pad = "SAME"
        elif type == "bilinear":
            filter = np.array([[[[.25]], [[.5]], [[.25]]],
                                    [[[.5]], [[1]], [[.5]]],
                                    [[[.25]], [[.5]], [[.25]]]])
            self.strides = [1, 2, 2, 1]
            self.pad = [[0, 0], [1, 1], [1, 1], [0, 0]]
        elif type == "linear":
            filter = np.array([[[[0]], [[.5]], [[.5]]],
                                    [[[.5]], [[1]], [[.5]]],
                                    [[[.5]], [[.5]], [[0]]]])
            self.strides = [1, 2, 2, 1]
            self.pad = [[0, 0], [1, 1], [1, 1], [0, 0]]
        elif type == "average":
            filter = np.array([[[[1 / 9]], [[1 / 9]], [[1 / 9]]],
                                    [[[1 / 9]], [[1 / 9]], [[1 / 9]]],
                                    [[[1 / 9]], [[1 / 9]], [[1 / 9]]]])
            self.strides = [1, 2, 2, 1]
            self.pad = [[0, 0], [1, 1], [1, 1], [0, 0]]
        else:
            raise Exception("In Conv2d_fixed select: border, bilinear, linear or average.")

        self.filter = np.repeat(filter.astype(np.float32), [out_channels], axis=3)

class Conv2DFixed(keras.layers.Layer, Filters):
    def __init__(self, kernel_type, out_channels):
        super(Conv2DFixed, self).__init__()
        Filters.__init__(self, kernel_type, out_channels)
        self.w = tf.Variable(initial_value=tf.constant(self.filter, dtype=tf.float32), trainable=False)
        self.bn = BatchNormalization()

    def call(self, inputs):
        channels = tf.nn.conv2d(inputs, self.w, strides=self.strides, padding=self.pad)
        norm = self.bn(channels)
        return norm

class Conv2DFixed_Transpose(keras.layers.Layer, Filters):
    def __init__(self, kernel_type, out_shape):
        super(Conv2DFixed_Transpose, self).__init__()
        Filters.__init__(self, kernel_type, out_shape[3])
        self.out_shape = out_shape
        self.w = tf.Variable(initial_value=tf.constant(self.filter, dtype=tf.float32), trainable=False)
        self.bn = BatchNormalization()

    def call(self, inputs):
        channels = tf.nn.conv2d_transpose(inputs, self.w, output_shape=self.out_shape, strides=self.strides,
                                         padding=self.pad)
        norm = self.bn(channels)
        return norm

class Conv2D_NA(keras.layers.Layer):
    def __init__(self, k_dim, output_channel, stride, padding='VALID', k_reg=None):
        super(Conv2D_NA, self).__init__()
        self.stride = stride
        self.padding = padding

        self.conv = Conv2D(filters=output_channel, kernel_size=(k_dim, k_dim), strides=(stride, stride),
                           padding=padding, kernel_regularizer=k_reg, use_bias=False)
        self.bn = BatchNormalization()

    def call(self, inputs):
        channels = self.conv(inputs)
        norm = self.bn(channels)
        return tf.nn.relu(norm)

def SNet_5L0(inputs, batch, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 1, Li
    n1Li = Conv2D_NA(k_dim=3, output_channel=2, stride=1, padding="SAME", k_reg=l2)(inputs)

    # - Level 2, Li
    n2Li = Conv2DFixed("bilinear", out_channels=2)(n1Li)
    n2Li = Conv2D_NA(k_dim=3, output_channel=4, stride=1, padding="SAME", k_reg=l2)(n2Li)

    # - Level 3, Li
    n3Li = Conv2DFixed("bilinear", out_channels=4)(n2Li)
    n3Li = Conv2D_NA(k_dim=3, output_channel=8, stride=1, padding="SAME", k_reg=l2)(n3Li)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=8)(n3Li)
    n4Li = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, M
    n5M = Conv2D_NA(k_dim=5, output_channel=128, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5M], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 64])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld])
    n3Ld = Conv2D_NA(k_dim=3, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2DFixed_Transpose("bilinear", [batch, 360, 640, 5])(n3Ld)

    # - Level 2, Ld
    n2Ld = concatenate([n2Li, n3Ld])
    n2Ld = Conv2D_NA(k_dim=3, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n2Ld)
    n2Ld = Conv2DFixed_Transpose("bilinear", [batch, 720, 1280, 5])(n2Ld)

    # - Level 1, Ld
    n1Ld = concatenate([n1Li, n2Ld])
    n1Ld = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(n1Ld)
    return n1Ld

def SNet_5L1(inputs, batch, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 1, Li
    n1Li = Conv2D_NA(k_dim=3, output_channel=2, stride=1, padding="SAME", k_reg=l2)(inputs)

    # - Level 2, Li
    n2Li = Conv2DFixed("bilinear", out_channels=2)(n1Li)
    n2Li = Conv2D_NA(k_dim=3, output_channel=4, stride=1, padding="SAME", k_reg=l2)(n2Li)

    # - Level 3, Li
    n3Li = Conv2DFixed("bilinear", out_channels=4)(n2Li)
    n3Li = Conv2D_NA(k_dim=3, output_channel=8, stride=1, padding="SAME", k_reg=l2)(n3Li)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=8)(n3Li)
    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 6, M
    n6M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n6M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld])
    n3Ld = Conv2D_NA(k_dim=3, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2DFixed_Transpose("bilinear", [batch, 360, 640, 5])(n3Ld)

    # - Level 2, Ld
    n2Ld = concatenate([n2Li, n3Ld])
    n2Ld = Conv2D_NA(k_dim=3, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n2Ld)
    n2Ld = Conv2DFixed_Transpose("bilinear", [batch, 720, 1280, 5])(n2Ld)

    # - Level 1, Ld
    n1Ld = concatenate([n1Li, n2Ld])
    n1Ld = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(n1Ld)
    return n1Ld

def SNet_5L2(inputs, batch, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 1, Li
    fixed_n1Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n1Li, inputs], axis=3)

    n1Li = Conv2D_NA(k_dim=3, output_channel=8, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 2, Li
    n2Li = Conv2DFixed("bilinear", out_channels=8)(n1Li)
    inputs_n2Li = tf.image.resize(input_fixed, [360, 640], antialias=False)
    n2Li = concatenate([n2Li, inputs_n2Li], axis=3)

    n2Li = Conv2D_NA(k_dim=3, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n2Li)

    # - Level 3, Li
    n3Li = Conv2DFixed("bilinear", out_channels=32)(n2Li)
    inputs_n3Li = tf.image.resize(input_fixed, [180, 320], antialias=False)
    n3Li = concatenate([n3Li, inputs_n3Li], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Li)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=32)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [90, 160], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [45, 80], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)
    n5M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n5M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld], axis=3)
    n3Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2DFixed_Transpose("bilinear", [batch, 360, 640, 32])(n3Ld)

    # - Level 2, Ld
    n2Ld = concatenate([n2Li, n3Ld], axis=3)
    n2Ld = Conv2D_NA(k_dim=3, output_channel=8, stride=1, padding="SAME", k_reg=l2)(n2Ld)
    n2Ld = Conv2DFixed_Transpose("bilinear", [batch, 720, 1280, 8])(n2Ld)

    # - Level 1, Ld
    n1Ld = concatenate([n1Li, n2Ld])
    n1Ld = Conv2D_NA(k_dim=3, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n1Ld)
    n1Ld = Conv2D(filters=2, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation=None, use_bias=False)(n1Ld)
    n1Ld = BatchNormalization()(n1Ld)
    n1Ld = tf.keras.activations.softmax(n1Ld)

    return n1Ld

def SNet_5L3(inputs, batch, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 1, Li
    fixed_n1Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n1Li, inputs], axis=3)

    n1Li = Conv2D_NA(k_dim=3, output_channel=8, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 2, Li
    n2Li = Conv2DFixed("bilinear", out_channels=8)(n1Li)
    inputs_n2Li = tf.image.resize(input_fixed, [360, 640], antialias=False)
    n2Li = concatenate([n2Li, inputs_n2Li], axis=3)

    n2Li = Conv2D_NA(k_dim=3, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n2Li)

    # - Level 3, Li
    n3Li = Conv2DFixed("bilinear", out_channels=32)(n2Li)
    inputs_n3Li = tf.image.resize(input_fixed, [180, 320], antialias=False)
    n3Li = concatenate([n3Li, inputs_n3Li], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Li)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=32)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [90, 160], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [45, 80], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)
    n5M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n5M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld], axis=3)
    n3Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2DFixed_Transpose("bilinear", [batch, 360, 640, 32])(n3Ld)

    # - Level 2, Ld
    n2Ld = concatenate([n2Li, n3Ld], axis=3)
    n2Ld = Conv2D_NA(k_dim=3, output_channel=8, stride=1, padding="SAME", k_reg=l2)(n2Ld)
    n2Ld = Conv2DFixed_Transpose("bilinear", [batch, 720, 1280, 8])(n2Ld)

    # - Level 1, Ld
    n1Ld = concatenate([n1Li, n2Ld])
    n1Ld = Conv2D_NA(k_dim=3, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n1Ld)
    n1Ld = Conv2D(filters=2, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation=None, use_bias=False)(n1Ld)
    n1Ld = BatchNormalization()(n1Ld)
    n1Ld = tf.keras.activations.softmax(n1Ld)

    return n1Ld

def SNet_4L(inputs, batch, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 2, Li
    fixed_n2Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n2Li, inputs], axis=3)

    n2Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 3, Li
    n3Li = Conv2DFixed("bilinear", out_channels=32)(n2Li)
    inputs_n3Li = tf.image.resize(input_fixed, [180, 320], antialias=False)
    n3Li = concatenate([n3Li, inputs_n3Li], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Li)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=32)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [90, 160], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [45, 80], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 6, M
    n6M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n6M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld], axis=3)
    n3Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2DFixed_Transpose("bilinear", [batch, 360, 640, 32])(n3Ld)

    # - Level 2, Ld
    n2Ld = concatenate([n2Li, n3Ld])
    n2Ld = Conv2D_NA(k_dim=5, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n2Ld)
    n2Ld = Conv2D(filters=2, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation=None, use_bias=False)(n2Ld)
    n2Ld = BatchNormalization()(n2Ld)
    n2Ld = tf.keras.activations.softmax(n2Ld)

    return n2Ld

def SNet_3L(dim, learn_reg=1e-2):
    batch = dim[0]
    inputs = Input(shape=dim[1:])
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 3, Li
    fixed_n3Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n3Li, inputs], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=32)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [90, 160], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [45, 80], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 6, M
    n6M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n6M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld])
    n3Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2D(filters=2, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation=None, use_bias=False)(n3Ld)
    n3Ld = BatchNormalization()(n3Ld)
    n3Ld = tf.keras.activations.softmax(n3Ld)

    return Model(inputs, n3Ld)

def SNet_3L_overfitting(dim, learn_reg=1e-2):
    batch = dim[0]
    inputs = Input(shape=dim[1:])

    # Left side = Li, Mid = M y right side = Ld
    # - Level 3, Li
    fixed_n3Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n3Li, inputs], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME")(input_fixed)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=64)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [90, 160], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=256, stride=1, padding="SAME")(n4Li)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=256)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [45, 80], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=512, stride=1, padding="SAME")(n5Li)

    # - Level 6, M
    n6M = Conv2D_NA(k_dim=5, output_channel=1024, stride=1, padding="SAME")(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n6M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=512, stride=1, padding="SAME")(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 512])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=256, stride=1, padding="SAME")(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 256])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld])
    n3Ld = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME")(n3Ld)
    n3Ld = Conv2D(filters=2, kernel_size=(1, 1), padding="SAME", activation=None, use_bias=False)(n3Ld)
    n3Ld = BatchNormalization()(n3Ld)
    n3Ld = tf.keras.activations.softmax(n3Ld)

    return Model(inputs, n3Ld)

def SNet_3L_plusplus(inputs, batch, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 3, Li
    fixed_n3Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n3Li, inputs], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=32)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [90, 160], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 3-4
    n34 = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Li)
    n34 = concatenate([n3Li, n34, input_fixed], axis=3)
    n34 = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n34)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [45, 80], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 6, M
    n6M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n6M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld, n34, input_fixed])
    n3Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(n3Ld)
    return n3Ld

def SNet_3L_p(inputs, batch, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 3, Li
    fixed_n3Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n3Li, inputs], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=32)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [90, 160], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 3-4
    n34 = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Li)
    n34 = concatenate([n3Li, n34], axis=3)
    n34 = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n34)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [45, 80], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 6, M
    n6M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n6M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld, n34])
    n3Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(n3Ld)
    return n3Ld

def SNet_3Lite(inputs, batch, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 3, Li
    fixed_n3Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n3Li, inputs], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=8, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=8)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [90, 160], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [45, 80], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 6, M
    n6M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n6M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld])
    n3Ld = Conv2D_NA(k_dim=5, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2D(filters=2, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(n3Ld)
    return n3Ld
