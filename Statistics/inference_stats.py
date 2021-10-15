from Networks.ModelManager import InferenceModel
from Data.DataManager import DataManager
from Statistics.StatsModel import InferenceStats
import cv2

"""
This script generates the metrics of the selected model
"""

if __name__ == '__main__':
    # Statistic Variables
    verbose = 1
    example = 2
    p = 0.01
    stats_type = "seg" # det or seg
    function = {1: "one_example", 2: "valid_set", 3: "A, IoU, P, R, F1", 4: "save images"}
    select_function = 3
    save_in = r"C:\Users\TTe_J\Downloads\resultados" # solo para "save images"
    tablefmt = "grid" # grid or latex

    # Net Variables
    # input_dims = (8, 720, 1280, 3)
    # input_dims = (8, 513, 1025, 3)
    # input_dims = (8, 360, 640, 3)
    input_dims = (8, 180, 320, 3)
    model = "SNet_3L"  # <========= models = HelperNetV1, ..V2, ..V3, SNet_5L2, .._4L, .._3L, .._3L_plusplus, .._3Lite, .._4, .._5, .._6, MgNet_0,
                        # UNetplusplus_3L
    output_type = "cls"  # regression = reg, classification = cls, regression + classficiation = reg+cls
    inference_type = "mask4seg" # bbox4reg, bbox4seg, mask4reg or mask4seg
    min_area = 3 # <= for bbox4reg
    neighbours = 3 # <= for bbox4reg
    start_epoch = 142  # <= numero de épocas que ya ha entrenado
    color_space = None # <= bgr=None, lab=44, yuv=82, hsv=40, hsl=52
    specific_weights = f"real_cls_bgr"
    weights_path = f'../Weights/{model}/{specific_weights}_epoch'

    # Results from csv
    from_csv = False
    csv_file = r'C:\Users\TTe_J\Downloads\csv_javi_formzero_fixed_archive.csv'

    # Data Variables
    inputs_rgb = [r'C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\First_microscope_all_320x180']
    input_type = "png" # png, jpg
    # original_size = (2988, 5312)
    original_size = (180, 320)
    # inputs_rgb = [r'C:\Users\TTe_J\Downloads\17-17-05']
    labels_class = ["binary"]
    label_type = "json" # mask or json
    # labels = ["b", "y", "o_s", "o_b"]
    label_size = (input_dims[1], input_dims[2], len(labels_class))
    background = True
    batch_size = 8
    valid_size = .10

    dm = DataManager(inputs_rgb, input_type, original_size, labels_class, label_size, background, valid_size, batch_size,
                     label_type, output_type)
    im = InferenceModel(model, input_dims, weights_path, start_epoch, output_type, inference_type, original_size,
                        min_area, neighbours)

    sm = InferenceStats(im, dm, output_type, stats_type, p)

    if function[select_function] == function[1]:
        sm.one_example(example, verbose=verbose)
    elif function[select_function] == function[2]:
        sm.set(color_space=color_space, tablefmt=tablefmt)
    elif function[select_function] == function[3]:
        sm.resume(model+"_"+specific_weights, color_space=color_space, from_csv=from_csv, csv_file=csv_file)
    elif function[select_function] == function[4]:
        counter = 0
        for idx in range(dm.batches_size["valid"] - 1):
            _example_xs = dm.batch_x(idx, "valid", color_space)
            _example_ys = dm.batch_y_bbox(idx, "valid") if stats_type == "det" else dm.batch_y(idx, "valid")
            _ys_hat = im.predict(_example_xs)
            for i in range(len(_example_xs)):
                if counter == 73:
                    print("hola")
                # cv2.imwrite(save_in + f"/{counter}_x.png", _example_xs[i]*255)
                # cv2.imwrite(save_in + f"/{counter}_y.png", _example_ys[i, ..., 0]*255)
                # cv2.imwrite(save_in + f"/{counter}_h.png", _ys_hat[i, ..., 0]*255)
                counter += 1
    else:
        print("Error selecting function!!")