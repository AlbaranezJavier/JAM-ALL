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
        # Managing directories
        self._constants()
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
        self.num_classes = label_size[2] + 1 if self.background else label_size[2]
        self.labels = labels
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

    # Input data
    @classmethod
    def _constants(self):
        self.CELL = "Cell_"
        self.IMGS = "/images/"
        self.JSONS = "/jsons/"
        self.NUMBER = "Cell Numbers"

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
                _img = cv2.rotate(_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
        if self.labels == ["binary"]:
            mask = np.zeros(tuple(list(self.original_size)+[2]), dtype="float32")
            with open(path, "r") as reader:
                data = json.load(reader)
                reader.close()
            for i in range(data[self.NUMBER]):
                x1 = int(data[self.CELL+str(i)]["x1"])
                x2 = int(data[self.CELL+str(i)]["x2"])
                y1 = int(data[self.CELL+str(i)]["y1"])
                y2 = int(data[self.CELL+str(i)]["y2"])
                mask[y1:y2, x1:x2, 0] = 1.
                mask[:, :, 1] = ((mask[:, :, 0] == 0)*1.).astype("float32")
            if self.original_size != self.label_size[0:2]:
                mask = cv2.resize(mask, (self.label_size[1], self.label_size[0]), cv2.INTER_NEAREST)
            return mask
        else:
            pass

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

    def x_idx(self, idx, step):
        """
        Get one example by index
        :param idx: example index
        :param step: training or valid
        :return: mask
        """
        _img = cv2.imread(self.X[step][idx])
        _rgb = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        return _img.reshape((1, _img.shape[0], _img.shape[1], _img.shape[2])) / 255., _rgb

    def y_idx(self, idx, step):
        """
        Get one example by index
        :param idx: example index
        :param step: trining or valid
        :return: mask
        """
        return self._label2mask(self.Y[step][idx], self.output_type)

    def _batch_division(self, set, batch_size):
        batch_idx = np.arange(0, len(set), batch_size)
        return batch_idx

    # Annotation management
    def _label2mask(self, data, output_type, original_size=(720, 1280)):
        """
        Generates a mask with the labeling data
        :param data: labeling
        :param output_type: boolean, true = mask between [0, 255] with radial gradient or false = mask between [0,1]
        :param original_size: original size mask labels
        :return: mask
        """
        img = np.ones([self.label_size[0], self.label_size[1], self.num_classes], dtype=np.uint8)
        for lab in range(self.label_size[2]):
            zeros = np.zeros(original_size, dtype=np.uint8)
            for idx in range(len(data[lab])):
                if output_type == "reg" or output_type == "reg+cls":
                    _x, _y, _w, _h = cv2.boundingRect(data[lab][idx])
                    _shape = zeros[_y:_y + _h, _x:_x + _w].shape
                    if _shape[0] // 2 != 0 and _shape[1] // 2 != 0:
                        cv2.fillConvexPoly(zeros, data[lab][idx], 255)
                        _v_grad = np.repeat(1 - np.abs(np.linspace(-0.9, 0.9, _shape[1], dtype=np.float16))[None],
                                            _shape[0], axis=0)
                        _h_grad = np.repeat(1 - np.abs(np.linspace(-0.9, 0.9, _shape[0], dtype=np.float16))[None],
                                            _shape[1], axis=0).T
                        _grad_mask = _v_grad * _h_grad
                        _grad_mask[_shape[0] // 2, _shape[1] // 2] = 1.0
                        zeros[_y:_y + _h, _x:_x + _w] = (zeros[_y:_y + _h, _x:_x + _w] * _grad_mask).astype(np.uint8)
                else:
                    cv2.fillConvexPoly(zeros, data[lab][idx], 1)
            zeros = cv2.resize(zeros, (self.label_size[1], self.label_size[0]), cv2.INTER_NEAREST)
            img[:, :, lab] = zeros.copy()
            if self.background:
                img[:, :, self.num_classes-1] *= np.logical_not(zeros)
        return img

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


    @classmethod
    def mask2vgg(self, masks, labels, names, sizes, save_path=None):
        """
        Generates labeling data from a mask
        :param masks: 1 to N channels
        :param labels: classes
        :param names: file name
        :param sizes: file size
        :param save_path: None = not saving, other case = saving as json
        :return: json format
        """
        file = {}
        for i in range(len(masks)):
            regions = []
            counter = 0
            mask = np.argmax(masks[i], axis=2)
            for m in range(masks[i].shape[2] - 1):
                contours, _ = cv2.findContours(np.uint8((mask == m) * 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in range(len(contours)):
                    # At least three points to form a polygon
                    countourX = []
                    countourY = []
                    if len(contours[c][:, :, 0]) > 2:
                        countourX = contours[c][:, :, 0][:, 0].tolist()
                        countourY = contours[c][:, :, 1][:, 0].tolist()

                    regions.append({
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": countourX,
                            "all_points_y": countourY
                        },
                        "region_attributes": {
                            "type": labels[m]
                        }
                    })
                    counter += 1
            file[names[i]] = {

                "filename": names[i],
                "size": sizes[i],
                "regions": regions,
                "file_attributes": {}
            }

        if save_path != None:
            json_file = json.dumps(file, separators=(',', ':'))
            with open(save_path, "w") as outfile:
                outfile.write(json_file)
                outfile.close()
        return file

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
