import time, sys
from Networks.ModelManager import TrainingModel, set_seeds
from Data.DataManager import DataManager
from Statistics.StatsModel import TrainingStats
from tensorflow_addons.optimizers import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from vit_keras import vit

'''
This script executes the training of the network.
'''

if __name__ == '__main__':
    set_seeds()
    # Net Variables
    model = "ViT_B"
    start_epoch = 0
    id_copy = "_cropped_v3_all_384x384"
    end_epoch = 100
    save_weights = True
    min_acc = 94
    specific_weights = "" + id_copy
    input_dims = (8, 384, 384, 3)
    lr = 1e-5

    tm = TrainingModel(nn=vit.vit_b32(image_size=384,
                                      activation='linear',
                                      pretrained=True,
                                      include_top=True,
                                      pretrained_top=False,
                                      classes=6),
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
                       logs_name=f"{model}/cls/Raabin/{input_dims[1]}x{input_dims[2]}/{lr}/{end_epoch}",
                       start_epoch=start_epoch)

    for epoch in range(start_epoch + 1, end_epoch + 1):
        # Train
        start_time = time.time()
        loss_train, lr = [], -1
        for batch_x, batch_y in tqdm(train, desc=f'Train_batch: {epoch}'):
            batch_x = tf.image.resize(batch_x, input_dims[1:3])
            loss, lr = tm.train_step(batch_x, batch_y, epoch)
            loss_train.append(loss)
        train_acc = tm.get_acc_categorical("train")
        # Test
        loss_valid = []
        for batch_x, batch_y in tqdm(test, desc=f'Test_batch: {epoch}'):
            batch_x = tf.image.resize(batch_x, input_dims[1:3])
            loss_valid.append(tm.valid_step(batch_x, batch_y))
        valid_acc = tm.get_acc_categorical("valid")

        # Saves the weights of the model if it obtains the best result in validation
        end_time = round((time.time() - start_time) / 60, 2)
        is_saved = tm.save_best(ts.data["best"], valid_acc, min_acc, epoch, end_epoch, save_weights)
        ts.update_values(epoch, is_saved, np.mean(loss_train), np.mean(loss_valid), train_acc, valid_acc, end_time, lr,
                         verbose=1)
