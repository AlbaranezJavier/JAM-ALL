import json, cv2, os
from pprint import pprint
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from copy import deepcopy
from glob import glob
from tqdm import tqdm
"""
This script manages the data for training
"""

class CellDatasetLoader:
    TRAIN = "/train"
    TEST = "/test"
    XS = "/xs"
    YS = "/ys"

    def __init__(self, data_path: str, k_fold: int, batch: int):
        self.batch = batch
        self.train_path = data_path + f"/{k_fold}_fold" + self.TRAIN + self.XS + "/*.npy"
        self.test_path = data_path + f"/{k_fold}_fold" + self.TEST + self.XS + "/*.npy"

    @classmethod
    def read_npy(cls, path):
        data = [np.load(path.decode("utf-8")) for path in path.numpy()]
        return np.array(data)

    @classmethod
    def _loader(cls, xs_path: str) -> tuple:
        image = tf.py_function(cls.read_npy, [xs_path], tf.float32)

        ys_path = tf.strings.regex_replace(xs_path, "xs", "ys")
        mask = tf.py_function(cls.read_npy, [ys_path], tf.float32)

        return image, mask

    def get_sets(self, seed: int = 123) -> object:
        train = tf.data.Dataset.list_files(self.train_path, seed=seed)
        train = train.batch(self.batch, drop_remainder=True).map(self._loader).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        test = tf.data.Dataset.list_files(self.test_path, seed=seed)
        test = test.batch(self.batch, drop_remainder=True).map(self._loader).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train, test


class CellDatasetCreator:
    README_JSON = "/readme.json"
    IMAGE_FORMATS = {"png", "jpg"}
    ANNOTATION_FORMATS = {"json"}
    TRAIN_TEST = ["train", "test"]
    template_readme = {"name": "",
                       "summary": "",
                       "cross_validation": {"k_fold": -1,
                                            "n_train_files": -1,
                                            "n_test_files": -1,
                                            "n_files": -1},
                       "x_values": {"format": "",
                                    "shape": ()},
                       "y_values": {"format": "",
                                    "shape": ()}
                       }

    @classmethod
    def _process_img(cls, path: str, destination: str, new_size: tuple) -> None:
        img_name = path.split("\\")[-1].split('.')[0]
        img = cv2.imread(path)
        # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = cv2.resize(img, tuple(reversed(new_size)))
        img = img.astype(np.float32)/255.
        np.save(destination + f"/{img_name}", img)

    @classmethod
    def _process_json(cls, path: str, destination: str, new_size: tuple) -> None:
        img_name = path.split("\\")[-1].split('.')[0]
        with open(path, "r") as reader:
            data = json.load(reader)
            reader.close()
        mask = np.zeros(tuple(list(new_size) + [2]), dtype="float32")
        for i in range(data["Cell Numbers"]):
            x1 = int(data["Cell_" + str(i)]["x1"])
            x2 = int(data["Cell_" + str(i)]["x2"])
            y1 = int(data["Cell_" + str(i)]["y1"])
            y2 = int(data["Cell_" + str(i)]["y2"])

            # Control overlapping and negative index
            x_reg1 = 0 if x1 < 0 else x1
            y_reg1 = 0 if y1 < 0 else y1
            region = np.zeros(new_size, dtype=np.float32)
            region[y_reg1:y2, x_reg1:x2] = 1.

            # Cone generation
            patch_size = (y2-y1, x2-x1)
            x_axis = np.linspace(-1, 1, patch_size[0])[:, None]
            y_axis = np.linspace(-1, 1, patch_size[1])[None, :]
            _grad_mask = 1 - np.sqrt(x_axis ** 2 + y_axis ** 2)
            _grad_mask = np.clip(_grad_mask, 0., 1.)
            _grad_mask[patch_size[0] // 2, patch_size[1] // 2] = 1.0

            # Apply cone inside limits
            x_hat1 = abs(x1) if x1 < 0 else 0
            x_hat2 = (patch_size[1])-(x2-new_size[1]) if x2 > new_size[1] else patch_size[1]
            y_hat1 = abs(y1) if y1 < 0 else 0
            y_hat2 = (patch_size[0])-(y2-new_size[0]) if y2 > new_size[0] else patch_size[0]

            region[y_reg1:y2, x_reg1:x2] = (region[y_reg1:y2, x_reg1:x2] * _grad_mask[y_hat1:y_hat2, x_hat1:x_hat2]).astype(np.float32)

            # Add region generated to final mask
            mask[:, :, 0] = region * (region > mask[:, :, 0]) + mask[:, :, 0] * (region <= mask[:, :, 0])
            mask[:, :, 1] = np.ones_like(mask[:, :, 0]) - mask[:, :, 0]
        np.save(destination + f"/{img_name}", mask)

    @classmethod
    def create(cls, xs_path: str, ys_replace: list, destination_path: str, k_folds: list, new_size: tuple, name: str,
               summary: str):
        # Create readme.json
        cross_validation = {"k_fold": len(k_folds),
                            "n_train_files": len(k_folds[0]["train"]["xs"]),
                            "n_test_files": len(k_folds[0]["test"]["xs"]),
                            "n_files": len(k_folds[0]["train"]["xs"]) + len(k_folds[0]["test"]["xs"])}
        x_values = {"format": xs_path.split(".")[-1],
                    "shape": cv2.imread(k_folds[0]["train"]["xs"][0]).shape}
        y_values = {"format": ys_replace[-1][1],
                    "shape": "json structure"}
        cls._add_readme(destination_path, name, summary, cross_validation, x_values, y_values)

        # Processing block
        queries_format = {"ys": y_values["format"], "xs": x_values["format"]}
        for xy in queries_format:
            # Coupling process type according to file type
            path_format = queries_format[xy]
            if path_format in cls.IMAGE_FORMATS:
                func = cls._process_img
            elif path_format in cls.ANNOTATION_FORMATS:
                func = cls._process_json
            else:
                raise ValueError()

            for k in range(len(k_folds)):
                for tt in cls.TRAIN_TEST:
                    [func(path, k_folds[k][tt][f"{xy[0]}_dest"], new_size) for path in
                     tqdm(k_folds[k][tt][xy], desc=f"k: {k}, set: {tt}, subset: {xy}")]

    @classmethod
    def _add_readme(cls, destination_path: str, name: str, summary: str, cross_validation: dict, x_values: dict,
                    y_values: dict):
        readme = deepcopy(cls.template_readme)
        readme["name"] = name
        readme["summary"] = summary
        readme["cross_validation"] = cross_validation
        readme["x_values"] = x_values
        readme["y_values"] = y_values

        json_file = json.dumps(readme, separators=(',', ':'))
        with open(destination_path + cls.README_JSON, 'w') as outfile:
            outfile.write(json_file)
            outfile.close()


class DataManager:
    # Constants
    README_JSON = "/readme.json"
    K_FOLD = "k_fold"
    XS_FORMAT = "xs_format"
    YS_FORMAT = "ys_format"
    SPLITS = 5
    TRAIN_TEST = ["train", "test"]
    XS_YS = ["xs", "ys"]
    template_kfold = {"train": {"xs": [],
                                "x_dest": "",
                                "ys": [],
                                "y_dest": ""},
                      "test": {"xs": [],
                               "x_dest": "",
                               "ys": [],
                               "y_dest": ""}
                      }


    @classmethod
    def loadDataset(cls, data_path: str, k_fold: int, batch: int) -> CellDatasetLoader:
        with open(data_path + cls.README_JSON, "r") as reader:
            data = json.load(reader)
            reader.close()
        data["cross_validation"][cls.K_FOLD] = f"{data['cross_validation'][cls.K_FOLD]}: {k_fold} <= selected"
        pprint(data)
        return CellDatasetLoader(data_path, k_fold, batch)

    @classmethod
    def createDataset(cls, xs_path: str, ys_replace: list, destination_path: str, new_size: tuple, name: str = "",
                      summary: str = ""):
        # Creating directories in destination
        if len(os.listdir(path=destination_path)) > 0:
            print("Destination_path is not empty, first delete all")
            exit()
        k_folds = []
        for i in range(cls.SPLITS):
            k_folds.append(deepcopy(cls.template_kfold))
            directory: str = f"{destination_path}/{i}_fold"
            os.mkdir(directory)
            for j in cls.TRAIN_TEST:
                directory_j: str = f"{directory}/{j}"
                os.mkdir(directory_j)
                for k in cls.XS_YS:
                    directory_k: str = f"{directory_j}/{k}"
                    k_folds[i][j][f"{k[0]}_dest"] = directory_k
                    os.mkdir(directory_k)

        # Splitting data by k-fold, train-test and xs-ys
        xs, ys = cls._get_files(xs_path, ys_replace)
        skf = KFold(n_splits=cls.SPLITS, random_state=123, shuffle=True)
        fold = 0
        for train_index, test_index in skf.split(xs, ys):
            k_folds[fold]["train"]["xs"] = xs[train_index]
            k_folds[fold]["test"]["xs"] = xs[test_index]
            k_folds[fold]["train"]["ys"] = ys[train_index]
            k_folds[fold]["test"]["ys"] = ys[test_index]
            fold += 1

        CellDatasetCreator.create(xs_path, ys_replace, destination_path, k_folds, new_size, name, summary)

    @classmethod
    def _get_files(cls, xs_path: str, ys_regex: list) -> tuple:
        xs = glob(xs_path)
        ys = []
        for x in xs:
            for y in ys_regex:
                x = x.replace(y[0], y[1])
            ys.append(x)
        return np.array(xs), np.array(ys)


if __name__ == '__main__':
    selection = 1
    if selection == 0:
        # create dataset example
        path = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\first_v3_all_320x180\images\*.png"
        y_replace = [["images", "jsons"], ["png", "json"]]
        destination = r"D:\Datasets\Raabin\first_v3_all_320x180"
        new_size = (180, 320)
        name = "first_v3_all_320x180"
        summary = "Dataset for probabilistic segmentation of regions of interest in images of cells, in particular " \
                  "white blood cells."
        dm = DataManager.createDataset(path, y_replace, destination, new_size, name, summary)
    elif selection == 1:
        # load dataset example
        data_path: str = r"D:\Datasets\Raabin\first_v3_all_320x180"
        k_fold: int = 0
        train, test = DataManager.loadDataset(data_path, k_fold).get_sets()
        import matplotlib.pyplot as plt
        for image, mask in train.take(200):
            plt.imshow(image)
            plt.show()
            plt.imshow(mask[..., 0])
            plt.show()
