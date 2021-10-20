import os, json, cv2, shutil, glob, time
import numpy as np
from sklearn.model_selection import train_test_split

"""
This script manages the data for training
"""


class DataManager():
    """
    This class contains all the functionality necessary to control the input/output data to the network.
    """

    def __init__(self, rgb_path, input_type, original_size, labels, label_size, background, valid_size, batch_size, labels_type,
                 output_type, seed=123, shuffle=True):
        # Constants
        self.CELL = "Cell_"
        self.IMGS = "/images/"
        self.JSONS = "/jsons/"
        self.NUMBER = "Cell Numbers"
        self.CLASSES_DENSE = ["classes_dense"]

        # Managing directories
        self.original_size = original_size
        self.input_type = input_type
        self.output_type = output_type
        self.labels_type = labels_type
        self.rgb_paths, self.gt_paths = self._getpaths(rgb_path)
        _X_train, _X_valid, _Y_train, _Y_valid = train_test_split(self.rgb_paths, self.gt_paths,
                                                                  test_size=valid_size,
                                                                  random_state=seed, shuffle=shuffle)
        self.data_size = {"train": len(_X_train), "valid": len(_X_valid)}
        self.X = {"train": _X_train, "valid": _X_valid}
        self.Y = {"train": _Y_train, "valid": _Y_valid}
        self.background = background
        self.labels = labels
        self.num_classes = label_size[2] + 1 if self.background else label_size[2]
        self.label_size = label_size
        # Managing batches
        self.batches = {"valid": self._batch_division(_X_valid, batch_size),
                        "train": self._batch_division(_X_train, batch_size)}
        self.batches_size = {"train": len(self.batches["train"]), "valid": len(self.batches["valid"])}

        # Print data info
        print(f'Data Info\n - Size: {len(self.rgb_paths)}')
        print(f' - Train: {self.data_size["train"]} y valid: {self.data_size["valid"]}')
        print(f' - Train batches: {self.batches_size["train"]}, valid batches: {self.batches_size["valid"]}')
        print(
            f' - Paths: {rgb_path}, train: {self._get_info(rgb_path, "train")}, valid: {self._get_info(rgb_path, "valid")}')


    def _getpaths(self, imgs_path):
        '''
        Obtains the addresses of the input data
        :param imgs_path: directory of the images
        :param labels_path: jsons with the gt
        :return: numpy arrays with the images paths and the labels paths
        '''
        # Variables
        imgs = []
        labels = []
        for i in range(len(imgs_path)):
            paths = glob.glob(imgs_path[i] + self.JSONS + "*." + self.labels_type)
            for json_path in paths:
                labels.append(json_path)
                img_name = json_path.split("\\")[-1][:-5]
                imgs.append(imgs_path[i]+self.IMGS+img_name+"."+self.input_type)
        return np.array(imgs), np.array(labels)

    # Management of image batches
    def batch_x(self, idx, step, color_space):
        """
        Load batch X
        :param idx: index of batch
        :param step: could be train or valid
        :param color_space: format of the image: hsv=40, hsl=52, lab=44, yuv=82 or bgr=None.
        :return: array of images with values [0 - 1]
        """
        x = []
        for path in self.X[step][self.batches[step][idx]:self.batches[step][idx + 1]]:
            _img = cv2.imread(path)
            if self.original_size != self.label_size[0:2]:
                # _img = cv2.rotate(_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                _img = cv2.resize(_img, (self.label_size[1], self.label_size[0]))
            _img = cv2.cvtColor(_img, color_space).astype('float32') if color_space is not None else _img.astype(
                'float32')
            _img /= 255.
            x.append(_img)
        return np.array(x)

    def batch_y(self, idx, step):
        """
        Load batch Y
        :param idx: index of batch
        :param step: could be train or valid
        :return: array of masks [0-1]
        """
        y = []
        for path in self.Y[step][self.batches[step][idx]:self.batches[step][idx + 1]]:
            img = self._get4json(path) if self.labels_type == "json" else self._get4mask(path)
            y.append(img)
        return np.array(y)

    def _get4json(self, path):
        mask = np.zeros(tuple(list(self.original_size) + [2]), dtype="float32")
        with open(path, "r") as reader:
            data = json.load(reader)
            reader.close()
        if self.labels == ["binary"]:
            for i in range(data[self.NUMBER]):
                x1 = int(data[self.CELL+str(i)]["x1"])
                x2 = int(data[self.CELL+str(i)]["x2"])
                y1 = int(data[self.CELL+str(i)]["y1"])
                y2 = int(data[self.CELL+str(i)]["y2"])
                mask[y1:y2, x1:x2, 0] = 1.
                mask[:, :, 1] = ((mask[:, :, 0] == 0)*1.).astype("float32")
            if self.original_size != self.label_size[0:2]:
                mask = cv2.resize(mask, (self.label_size[1], self.label_size[0]), cv2.INTER_NEAREST)
        if self.labels == ["binary_dense"]:
            mask = np.zeros(tuple(list(self.original_size) + [2]), dtype="float32")
            for i in range(data[self.NUMBER]):
                x1 = int(data[self.CELL+str(i)]["x1"])
                x2 = int(data[self.CELL+str(i)]["x2"])
                y1 = int(data[self.CELL+str(i)]["y1"])
                y2 = int(data[self.CELL+str(i)]["y2"])
                _shape = mask[y1:y2, x1:x2, 0].shape
                if _shape[0] // 2 != 0 and _shape[1] // 2 != 0:
                    region = np.zeros(self.original_size, dtype=np.float32)
                    region[y1:y2, x1:x2] = 1.
                    _v_grad = np.repeat(1 - np.abs(np.linspace(-1., 1., _shape[1], dtype=np.float16))[None],
                                        _shape[0], axis=0)
                    _h_grad = np.repeat(1 - np.abs(np.linspace(-1., 1., _shape[0], dtype=np.float16))[None],
                                        _shape[1], axis=0).T
                    _grad_mask = _v_grad * _h_grad
                    _grad_mask[_shape[0] // 2, _shape[1] // 2] = 1.0
                    region[y1:y2, x1:x2] = (region[y1:y2, x1:x2] * _grad_mask).astype(np.float32)
                    mask[:, :, 0] = region * (region > mask[:, :, 0]) + mask[:, :, 0] * (region <= mask[:, :, 0])
                    mask[:, :, 1] = np.ones_like(mask[:, :, 0]) - mask[:, :, 0]
            if self.original_size != self.label_size[0:2]:
                mask = cv2.resize(mask, (self.label_size[1], self.label_size[0]), cv2.INTER_NEAREST)
        elif self.labels == ["classes_cnn"]:
            classes = {"Artifact": 0, "Burst": 1, "Eosinophil": 2, "Lymphocyte": 3, "Monocyte": 4, "Neutrophil": 5,
                       "Large Lymph": 3, "Small Lymph": 3, "Band": 5, "Meta": 5}
            mask = np.zeros((1, 1, 6))
            mask[:, :, classes[data["Label"]]] = 1.
        elif self.labels == ["classes_dense"]:
            classes = {"Artifact": 0, "Burst": 1, "Eosinophil": 2, "Lymphocyte": 3, "Monocyte": 4, "Neutrophil": 5,
                       "Large Lymph": 3, "Small Lymph": 3, "Band": 5, "Meta": 5}
            mask = np.zeros(6)
            mask[classes[data["Label"]]] = 1.
        return mask


    def _get4mask(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = np.zeros(tuple(list(img.shape)+[2]), dtype="float32")
        mask[:, :, 0] = ((img > 125)*1.).astype("float32")
        mask[:, :, 1] = ((img <= 125)*1.).astype("float32")
        mask = cv2.resize(mask, (self.label_size[1], self.label_size[0]), cv2.INTER_NEAREST)
        return mask

    def batch_y_bbox(self, idx, step):
        """
        Load batch Y
        :param idx: index of batch
        :param step: could be train or valid
        :return: array of labels
        """
        bboxs = []
        for labels in self.Y[step][self.batches[step][idx]:self.batches[step][idx + 1]]:
            bbox = []
            for c in range(len(labels)):
                for polygon in labels[c]:
                    _x, _y, _w, _h = cv2.boundingRect(polygon)
                    bbox.append([c, [_x, _y], [_x+_w, _y+_h]])
            bboxs.append(bbox)
        return bboxs

    def _batch_division(self, set, batch_size):
        batch_idx = np.arange(0, len(set), batch_size)
        return batch_idx

    # Annotation management
    def prediction2mask(self, prediction):
        """
        From prediction to mask
        :param prediction: prediction [0,1]
        :return: masks
        """

        if self.output_type == "cls":
            img = np.ones([self.label_size[0], self.label_size[1], self.num_classes], dtype=np.uint8)
            _idx_masks = np.argmax(prediction, axis=2)
            for lab in range(self.label_size[2]):
                img[..., lab] = ((_idx_masks == lab) * 1).astype(np.uint8)
            return img
        elif self.output_type == "reg":
            img = (prediction > 0)*1
            return img

    # Info data for training
    def _get_info(self, directories, step):
        """
        Count the number of images for each directory, differentiating between training and validation.
        :param directories: directories
        :param step: valid or train
        :return: train array, valid array
        """
        counter = [0 for i in directories]
        for path in self.X[step]:
            img_dir = path.split('/')[0]
            for i in range(len(directories)):
                if img_dir == directories[i]:
                    counter[i] += 1
        return counter
