import glob, cv2, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def read_json(path):
    path = path.numpy().decode("utf-8")
    with open(path, "r") as reader:
        data = json.load(reader)
        reader.close()
    mask = np.zeros((180, 320, 2), dtype="float32")
    for i in range(data["Cell Numbers"]):
        x1 = int(data["Cell_" + str(i)]["x1"])
        x2 = int(data["Cell_" + str(i)]["x2"])
        y1 = int(data["Cell_" + str(i)]["y1"])
        y2 = int(data["Cell_" + str(i)]["y2"])
        _shape = mask[y1:y2, x1:x2, 0].shape
        if _shape[0] // 2 != 0 and _shape[1] // 2 != 0:
            region = np.zeros((180, 320), dtype=np.float32)
            region[y1:y2, x1:x2] = 1.
            x_axis = np.linspace(-1, 1, _shape[0])[:, None]
            y_axis = np.linspace(-1, 1, _shape[1])[None, :]

            _grad_mask = 1 - np.sqrt(x_axis ** 2 + y_axis ** 2)
            _grad_mask = np.clip(_grad_mask, 0., 1.)
            _grad_mask[_shape[0] // 2, _shape[1] // 2] = 1.0

            region[y1:y2, x1:x2] = (region[y1:y2, x1:x2] * _grad_mask).astype(np.float32)
            mask[:, :, 0] = region * (region > mask[:, :, 0]) + mask[:, :, 0] * (region <= mask[:, :, 0])
            mask[:, :, 1] = np.ones_like(mask[:, :, 0]) - mask[:, :, 0]
    return mask

def read_npy_file(item):
    data = np.load(item.numpy().decode("utf-8"))
    return data

class myParser:
    @classmethod
    def parse_image(cls, img_path: str) -> tuple:
        """Load an image and its annotation (mask) and returning
        a dictionary.

        Parameters
        ----------
        img_path : str
            Image (not the mask) location.

        Returns
        -------
        dict
            Dictionary mapping an image and its annotation.
        """


        # image = tf.image.decode_png(image, channels=3)
        # image = tf.image.convert_image_dtype(image, tf.float32)/255.
        # img_path = tf.strings.regex_replace(img_path, "xs", "xs")
        image = tf.py_function(read_npy_file, [img_path], tf.float32)
        # print(img_path)
        # mask_path = tf.strings.regex_replace(img_path, "png", "npy")
        # mask_path = tf.strings.regex_replace(mask_path, "20161024_145318", "example")
        # mask = tf.py_function(read_npy_file, [mask_path], tf.uint8)
        mask_path = tf.strings.regex_replace(img_path, "xs", "ys")
        mask = tf.py_function(read_npy_file, [mask_path], tf.float32)


        return image, mask

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    # input_image = tf.image.resize(datapoint['image'], (180, 320))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (180, 320))

    # if tf.random.uniform(()) > 0.5:
    #     input_image = tf.image.flip_left_right(input_image)
    #     input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(datapoint['image'], input_mask)

    return input_image, input_mask

if __name__ == '__main__':
    train_dataset = tf.data.Dataset.list_files(r"D:\Datasets\Raabin\first_v3_all_320x180\0_fold\test\xs\*.npy", seed=123)
    train_dataset = train_dataset.map(myParser.parse_image)
    # train = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # train = train.shuffle(10, seed=123)
    # train = train.repeat()
    # train = train.batch(8)
    # train = train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # print(train)

    for image, mask in train_dataset.take(1):
        plt.imshow(image)
        plt.show()
        plt.imshow(mask[..., 0])
        plt.show()