import time, sys
from Networks.ModelManager import TrainingModel, set_seeds
from Data.DataManager import DataManager
from Statistics.StatsModel import TrainingStats
from tensorflow_addons.optimizers import AdamW
from Networks.ViT import SLICO_ViT, SLICOprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

'''
This script executes the training of the network.
'''

if __name__ == '__main__':
    set_seeds()
    # Net Variables
    model = "SLICO_ViT"
    start_epoch = 0
    id_copy = "_cropped_v3_all_256x256"
    end_epoch = 500
    save_weights = True
    min_acc = 94
    specific_weights = "" + id_copy
    input_dims = (32, 256, 256, 3)
    patch_size = 16
    projection_dim = 256
    num_patches = (input_dims[1] // patch_size) ** 2
    lr = 1e-5

    tm = TrainingModel(nn=SLICO_ViT(input_shape=input_dims,
                                    num_classes=6,
                                    projection_dim=projection_dim,
                                    num_patches=num_patches,
                                    transformer_layers=6,
                                    num_heads=4,
                                    transformer_units=[projection_dim, ],
                                    mlp_head_units=[512]),
                       weights_path=f'../Weights/{model}/{specific_weights}_epoch',
                       start_epoch=start_epoch,
                       optimizer=AdamW(learning_rate=lr, weight_decay=1e-6),
                       schedules={},
                       loss_func="categorical_crossentropy_true",
                       metric_func="categorical_accuracy")

    # Data Variables
    train, test = DataManager.loadDataset(
        data_path=r"D:\Datasets\Raabin\cropped_v3_all_512x512",
        k_fold=0,
        batch=input_dims[0]
    ).get_sets(seed=123)

    # Statistics
    ts = TrainingStats(model_name=model + id_copy,
                       specific_weights=specific_weights,
                       logs_tensorboard=f"{model}/cls/Raabin/batch_{input_dims[0]}/{input_dims[1]}x{input_dims[2]}/AdamW/{lr}/{end_epoch}",
                       start_epoch=start_epoch)

    for epoch in range(start_epoch + 1, end_epoch + 1):
        # Train
        start_time = time.time()
        loss_train, lr = [], -1
        for batch_x, batch_y in tqdm(train, desc=f'Train_batch: {epoch}'):
            batch_x = tf.image.resize(batch_x, input_dims[1:3])
            patches, positions = SLICOprocess(batch_x.numpy(), region_size=patch_size, ruler=10., iterations=50,
                                              num_patches=num_patches, projection_dim=projection_dim)
            loss, lr = tm.train_step({'patches': patches, 'positions': positions}, batch_y, epoch)
            loss_train.append(loss)
        train_acc = tm.get_acc_categorical("train")
        # Test
        loss_valid = []
        for batch_x, batch_y in tqdm(test, desc=f'Test_batch: {epoch}'):
            batch_x = tf.image.resize(batch_x, input_dims[1:3])
            patches, positions = SLICOprocess(batch_x.numpy(), region_size=patch_size, ruler=10., iterations=50,
                                              num_patches=num_patches, projection_dim=projection_dim)
            loss_valid.append(tm.valid_step({'patches': patches, 'positions': positions}, batch_y))
        valid_acc = tm.get_acc_categorical("valid")

        # Saves the weights of the model if it obtains the best result in validation
        end_time = round((time.time() - start_time) / 60, 2)
        is_saved = tm.save_best(ts.data["best"], valid_acc, min_acc, epoch, end_epoch, save_weights)
        ts.update_values(epoch, is_saved, np.mean(loss_train), np.mean(loss_valid), train_acc, valid_acc, end_time, lr,
                         verbose=1)
