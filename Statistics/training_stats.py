from Statistics.StatsModel import TrainingStats

"""
This script generates the metrics of the training process
"""

if __name__ == '__main__':
    # Variables
    model_name = "ViT_B_blocked"
    id_copy = "_cropped_v3_all_384x384"
    specific_weights = "_cropped_v3_all_384x384_epoch_100"

    ts = TrainingStats(model_name+id_copy, specific_weights)

    ts.print_data(y_lim_epoch=[50, 99.99], x_lim_loss=[0, 100], title=model_name+id_copy)
