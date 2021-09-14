from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from preprocess import read_data, splits_to_loaders
import os
import json
import pickle
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from cnn import CNNClog
from lstm import LSTMClog
from sklearn.model_selection import ParameterGrid


# DeepWrapper is a wrapper class that offers some useful functions
# (fit, predict) instead of implementing them in the "main" code
# (CNNWrapper and LSTMWrapper wrap CNN and LSTM pytorch models respectively)
class DeepWrapper:
    def __init__(self):
        raise Exception("Can't instantiate DeepWrapper directly!")

    def fit(self, dl, epochs=5000):
        self.loss_func = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.model.parameters())

        for epoch in tqdm(range(epochs)):
            for i, (X, y, _) in enumerate(dl):
                self.opt.zero_grad()
                y_pred = self.model(X)
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.opt.step()
    
    def predict(self, dl):
        # get predictions for each window, then use soft majority voting to get overall winner
        # (using pandas dataframes to handle the aggregation of the votes)
        self.model.eval()
        
        X, y, f = next(iter(dl))
        f = [ os.path.basename(a) for a in f ]

        y_pred = torch.softmax(self.model(X), dim=1).cpu().detach().numpy()
        df = pd.DataFrame(data={
            "file": f,
            "y_true": y.cpu().detach().numpy(),
            "pred_0": y_pred[:,0],
            "pred_1": y_pred[:,1],
            "pred_2": y_pred[:,2]
        })
        y_pred_all = df.groupby("file").mean()[["pred_0", "pred_1","pred_2"]].values.argmax(axis=1)
        y_true_all = df.groupby("file").mean()["y_true"].values

        self.model.train()
        
        return y_pred_all, y_true_all

class CNNWrapper(DeepWrapper):
    def __init__(self, channel_num, seq_len, out_classes):
        self.model = CNNClog(channel_num, seq_len, out_classes)

class LSTMWrapper(DeepWrapper):
    def __init__(self, channels_num, seq_size, out_classes):
        self.model = LSTMClog(channels_num, seq_size, out_classes)

# some utility function to keep track of the current
# state of execution (i.e. what has currently been processed)
def processed_list(tested_file):
    if not os.path.isfile(tested_file):
        return []
    with open(tested_file) as f:
        processed = json.load(f)
    return list(map(int,processed.keys()))

def update_processed(k, v, tested_file):
    if not os.path.isfile(tested_file):
        current = {}
    else:
        with open(tested_file) as f:
            current = json.load(f)
    current[k] = v
    with open(tested_file, "w") as f:
        json.dump(current, f)
    
if __name__ == "__main__":
    dataset, labels, filenames = read_data("data")

    # define some files that will be used as a support
    experiment = "experiment_name" # experiment name (unique Id)
    tested_file = f"tested_{experiment}.json" # this file will contain the results of the experiments
    confs_file = f"allconfs_{experiment}.pkl" # this file contains a list of all parameter configurations to be tested

    if not os.path.isfile(confs_file):
        # if there is no configuration file containing a list of all
        # parameters to be tried, a new one is generated 
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # performing both k-fold cross-validation and hold out
        folds = [ ("kfold", v) for v in list(skf.split(dataset, labels)) ]+ \
                [ ("tt", train_test_split(list(range(len(labels))), train_size=300, shuffle=False)) ]

        # model "agnostic" list of parameters to be configured
        # (here additional model-specific parameters could be added
        # if needed). 
        models_grid = {
            "model": ["cnn", "lstm"], 
            "random_state": [42],
            "chunk_size": [3729, 100], # length of each chunk of cycle to be used (3729 => use full cycle, 100 => use 100 seconds chunks)
            "fold": folds
        }
        confs = list(ParameterGrid(models_grid))
        
        with open(confs_file, "wb") as f:
            pickle.dump(confs, f)

    else: # if the config file is already available (e.g. because of a previous (partial) run), load it instead
        with open(confs_file, "rb") as f:
            confs = pickle.load(f)
        
    # here we decide whether we will be using CPUs or GPUs
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    for conf_id, config in enumerate(confs):
        # `tested_file` contains a list of previously performed experiments.
        # so that if we need to stop and restart the script, we have a state of
        # progress of the experiments (&& we avoid re-running those for which
        # we already have results)
        if conf_id in processed_list(tested_file):
            print("Skipping", conf_id, "(already processed)")
            continue # already processed this one
        
        model_type = config["model"]
        random_state = config["random_state"]
        chunk_size = config["chunk_size"]
        fold_type, stuff = config["fold"]
        flip_axes = model_type == "cnn" # the CNN model requires inverting the axes of the dataset
        
        print("Training:", model_type, random_state, chunk_size, fold_type)

        train_index, test_index = stuff

        X_train, y_train, fname_train = dataset[train_index], labels[train_index], filenames[train_index]
        X_test, y_test, fname_test = dataset[test_index], labels[test_index], filenames[test_index]

        dl_train, dl_test = splits_to_loaders(X_train, X_test, y_train, y_test, fname_train, fname_test, chunk_size=chunk_size, flip_axes=flip_axes, device=device)

        if model_type == "cnn":
            model = CNNWrapper(dl_train.dataset[0][0].shape[0], chunk_size, 3)
        elif model_type == "lstm":
            model = LSTMWrapper(dl_train.dataset[0][0].shape[1], dl_train.dataset[0][0].shape[0], 3)
        else:
            raise Exception("Unknown model")

        model.model.to(device)
        model.fit(dl_train)
        y_pred, y_true = model.predict(dl_test)

        result = {
            "model_info": {
                "model": model_type,
                "random_state": random_state,
                "chunk_size": chunk_size,
                "fold": fold_type
            },
            "performance": {
                "f1": list(f1_score(y_true, y_pred, average=None)),
                "accuracy": accuracy_score(y_true, y_pred)                
            }
        }
        print(result)
        update_processed(conf_id, result, tested_file)
