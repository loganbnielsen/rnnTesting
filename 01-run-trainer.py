from Trainer import Trainer
from Trainer import all_state_metrics

from AnalyzeExperimentHelpers import plot_experiment_results

import configs

from os import path
import pickle


def train(TRAIN_CONFIGS, GRU_CONFIGS):
    nTrial = TRAIN_CONFIGS.get("trials")
    for trial_num in range(nTrial):
        print(trial_num)
        if Trainer.pickled_exists(TRAIN_CONFIGS, trial_num):
            print("trial already ran.")
            pass
        else:
            trainer = Trainer(TRAIN_CONFIGS, GRU_CONFIGS)
            trainer.fit
            trainer.pickle_save(trial_num)
            trainer.gen_relative_graphs(trial_num, freq=10, pause=False)
            trainer.save_train_test_metrics(trial_num)
    
    plot_experiment_results(TRAIN_CONFIGS, "Relative Spacing", save=True)



if __name__ == "__main__":
    train(configs.TRAIN_CONFIGS, configs.GRU_CONFIGS)

