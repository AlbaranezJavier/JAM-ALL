import time, sys
from Networks.ModelManager import TrainingModel
from Data.DataManager import DataManager
from Statistics.StatsModel import TrainingStats
from Networks.SNet import *
from tensorflow.keras.optimizers import RMSprop, Adam
from tqdm import tqdm

'''
This script executes the training of the network.
'''

if __name__ == '__main__':
    # Net Variables
    model = "SNet_3L_overfitting"
    learn_opt, learn_reg = 1e-3, 1e-2
    start_epoch = 0  # <= number epoch trained
    id_copy = "_prob_v3"  # <= logs version? "" => main
    color_space = 44  # <= bgr=None, lab=44, yuv=82, hsv=40, hsl=52
    end_epoch = start_epoch + 100
    save_weights = True
    min_acc = 99.75
    specific_weights = "real" + id_copy
    input_dims = (16, 180, 320, 3)

    tm = TrainingModel(nn=locals()[model](input_dims, learn_reg),
                       weights_path=f'Weights/{model}/{specific_weights}_epoch',
                       start_epoch=start_epoch,
                       optimizer=Adam(learn_opt),
                       loss_func="categorical_crossentropy",
                       metric_func="categorical_accuracy")

    # Data Variables
    train, test = DataManager.loadDataset(
        data_path=r"D:\Datasets\Raabin\first_v3_all_320x180",
        k_fold=0,
        batch=input_dims[0]
    ).get_sets(seed=123)

    # Statistics
    ts = TrainingStats(model + id_copy, specific_weights, start_epoch)

    for epoch in range(start_epoch + 1, end_epoch + 1):
        # Train
        start_time = time.time()
        loss_value = 0
        for batch_x, batch_y in tqdm(train, desc=f'Train_batch: {epoch}'):
            loss_value += tm.train_step(batch_x, batch_y)
        train_acc = tm.get_acc("train")
        # Test
        for batch_x, batch_y in tqdm(test, desc=f'Test_batch: {epoch}'):
            tm.valid_step(batch_x, batch_y)
        valid_acc = tm.get_acc("valid")

        # Saves the weights of the model if it obtains the best result in validation
        end_time = round((time.time() - start_time) / 60, 2)
        is_saved = tm.save_best(ts.data["best"], valid_acc, min_acc, epoch, end_epoch, save_weights)
        ts.update_values(epoch, is_saved, loss_value, train_acc, valid_acc, end_time, verbose=1)
