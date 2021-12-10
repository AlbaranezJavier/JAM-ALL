import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import cv2
import random


def mlp(x, hidden_units, dropout_rate):
  for units in hidden_units:
    x = layers.Dense(units, activation=tf.nn.gelu)(x)
    x = layers.Dropout(dropout_rate)(x)
  return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images,
                                           sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, self.patch_size, self.patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding="VALID")
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def centroid(idx_array: np.ndarray) -> tuple:
    length = len(idx_array[0])
    return np.sum(idx_array[0])//length, np.sum(idx_array[1])//length

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return pos_encoding.astype(np.float32)[0]

def SLICOprocess(imgs: np.ndarray, region_size: int, ruler: float, iterations: int, max_labels: int) -> tuple:
    patches_batch, positions_batch = [], []
    for img in imgs:
        slic = cv2.ximgproc.createSuperpixelSLIC(img, algorithm=cv2.ximgproc.SLICO, region_size=region_size, ruler=ruler)
        slic.iterate(iterations)
        label_slic = slic.getLabels()
        # puede haber más parches de los esperados
        num_labels = np.max(label_slic)
        list_labels = np.array(range(num_labels))
        if num_labels > max_labels:
            list_labels = np.delete(list_labels, random.sample(range(num_labels), num_labels - max_labels))
        # codifica las posiciones (x,y) y obtiene los parches
        pos_encoding = positional_encoding(position=max_labels, d_model=max_labels // 2)
        patches, positions = [], []
        for l in list_labels:
            idx_label = np.where(label_slic == l)
            y, x = centroid(idx_label)
            # Aprovecho el loop para adaptar los parches a un tamaño fijo
            patch = np.zeros((max_labels, 3))
            temp_patch = np.array([img[y, x, :] for y, x in list(zip(idx_label[0], idx_label[1]))])
            if len(temp_patch) > max_labels:
                patch = np.delete(temp_patch, random.sample(range(len(temp_patch)), len(temp_patch) - max_labels), axis=0)
            else:
                patch[:len(temp_patch)] = temp_patch
            patches.append(patch)
            positions.append(np.concatenate((pos_encoding[x], np.flip(pos_encoding[y]))))
        # Tengo que rellenar con 0s tanto parches como posiciones
        for empt in range(len(list_labels), max_labels):
            patches.append(np.zeros((max_labels, 3)))
            positions.append(np.zeros(max_labels))
        patches_batch.append(np.array(patches))
        positions_batch.append(np.array(positions))
    return np.array(patches_batch), np.array(positions_batch)


def ViT(input_shape, num_classes, patch_size, num_patches, projection_dim, transformer_layers, num_heads, transformer_units, mlp_head_units):
    inputs = layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Transformer block
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)

        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        encoded_patches = layers.Add()([x3, x2])

    # [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify
    logits = layers.Dense(num_classes)(features)

    model = keras.Model(inputs, logits)
    return model

def SLICO_ViT(input_shape, num_classes, projection_dim, transformer_layers, num_heads, transformer_units, mlp_head_units):
    patches = layers.Input(shape=input_shape[1:], batch_size=input_shape[0], name="patches")
    patches_reshape = tf.reshape(patches, [input_shape[0], input_shape[1], input_shape[2]*input_shape[3]])
    projection = layers.Dense(units=projection_dim)(patches_reshape)

    positions = layers.Input(shape=input_shape[1:-1], batch_size=input_shape[0], name="positions")
    encoded_patches = projection + positions

    # Transformer block
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)

        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        encoded_patches = layers.Add()([x3, x2])

    # [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify
    logits = layers.Dense(num_classes)(features)

    model = keras.Model([patches, positions], logits)
    return model

def ViT_from_h5(path: str):
    return keras.models.load_model(path)
