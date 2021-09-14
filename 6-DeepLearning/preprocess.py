import pandas as pd
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class CycleDataset(Dataset):
    """
    - X: dataset
    - y: labels
    - filenames: list of filenames (used for the reconstruction given the chunks)
    - labels: list of labels (the order is used for the numerical encoding)
    - flip_axes: boolean, whether to flip 1st and 2nd axes in X (required by pytorch's CNN layer)
    - device: device to load the data to (cuda or cpu)
    """
    def __init__(self, X, y, filenames, labels, flip_axes=False, device=torch.device("cpu")):
        super().__init__()
        
        if flip_axes:
            self.X = torch.tensor(X.swapaxes(1,2)).float()
        else:
            self.X = torch.tensor(X).float()
        self.y = torch.tensor(list(map(labels.index, y)))
        self.filenames = filenames
        
        self.X = self.X.to(device)
        self.y = self.y.to(device)
        
        assert self.X.shape[0] == self.y.shape[0] and self.X.ndim == 3 and self.y.ndim == 1
    
    def __getitem__(self, i):
        return self.X[i], self.y[i], self.filenames[i]
    
    def __len__(self):
        return self.X.shape[0]

def chunkize(X, y, fnames, chunk_size=100):
    out = [ (b, l, f) for a,l,f in zip(X, y, fnames) for b in np.split(a, list(range(chunk_size, a.shape[0], chunk_size))) if b.shape[0] == chunk_size ]
    
    X_chunks = np.array([ x[0] for x in out ])
    y_chunks = np.array([ x[1] for x in out ])
    f_chunks = np.array([ x[2] for x in out ])
    return X_chunks, y_chunks, f_chunks

def read_data(datadir):
    signals_file = os.path.join(datadir, "signals", "signals-file.txt")
    labels_file = os.path.join(datadir, "labels", "labels-file.txt")
    cycles_file_pattern = os.path.join(datadir, "cycles", "engine-model", "cycles-naming-pattern-*.csv")
      
    with open(signals_file) as f:
        # load list of signals to be kept
        signals = [ line.strip() for line in f.readlines() ]

    with open(labels_file) as f:
        # build dictionary of { filename: label } 
        all_labels = dict([ line.strip().split(",") for line in f.readlines() ])

    labels = [] # labels, sorted in the same order as elements in dataset
    dataset = [] # dataset, as a list of matrices (one matrix per cycle)

    filenames = [] # keeping track of the name of each cycle (used when splitting into chunks for majority voting)

    with tqdm(sorted(glob(cycles_file_pattern))) as bar:
        for fname in bar:
            df = pd.read_csv(fname, skiprows=[1], usecols=signals)
            bname = os.path.basename(fname)

            if bname not in all_labels:
                continue

            labels.append(all_labels[bname])
            dataset.append(df.values)
            filenames.append(fname)
    
    # this is the length of the shortest track to be processed,
    # we will trim all cycles to the same length for convenience
    # (this means discarding some initial samples, not a big deal)
    # min: 3729
    # max: 3750
    # discarding at most the initial 21 seconds of each cycle
    min_len = min(map(len,dataset))
    
    dataset = np.array([ x[-min_len:] for x in dataset ])
    labels = np.array(labels)
    filenames = np.array(filenames)
    return dataset, labels, filenames

def splits_to_loaders(X_train, X_test, y_train, y_test, fname_train, fname_test, chunk_size=100, batch_size=-1, flip_axes=False, device=torch.device("cpu")):
    """
    Function that loads the data and builds DataLoaders for training pytorch models
    
    Arguments:
    - X_train, X_test, y_train, y_test: numpy arrays containing the respective data 
    - chunk_size: number of chunks into which each cycle should be converted (any exceeding portion
      of the signal (of length < chunk_size) will be discarded)
    - batch_size: batch size for the DataLoaders returned
    - random_state: random state to be used for the train/valid/test split (for reproducibility)
    - flip_axes: change the order of 1st and 2nd axis when creating the datasets (needed for the cnn)
    - device: the device to be used to store the data
    
    Returns:
    - dl_train: DataLoader for the training set
    - dl_test: DataLoader for the test set
    """

    # build chunks
    X_train_chunks, y_train_chunks, f_train_chunks = chunkize(X_train, y_train, fname_train, chunk_size)
    X_test_chunks, y_test_chunks, f_test_chunks = chunkize(X_test, y_test, fname_test, chunk_size)
    
    # normalize using standard scaler ("fit" on train, "transform" on test)
    ss = StandardScaler()
    X_train_chunks_std = np.array(np.vsplit(ss.fit_transform(np.vstack(list(X_train_chunks))), X_train_chunks.shape[0]))
    X_test_chunks_std = np.array(np.vsplit(ss.transform(np.vstack(list(X_test_chunks))), X_test_chunks.shape[0]))
    
    classes = ['red','yellow','green'] # order to be used for the classes (TODO: pass this as an argument from caller)

    # build dataset and dataloader for all splits
    ds_train = CycleDataset(X_train_chunks_std,  y_train_chunks, f_train_chunks, classes, flip_axes, device)
    dl_train = DataLoader(ds_train, batch_size=batch_size if batch_size > 0 else len(ds_train), shuffle=True)

    ds_test = CycleDataset(X_test_chunks_std,  y_test_chunks, f_test_chunks, classes, flip_axes, device)
    dl_test = DataLoader(ds_test, batch_size=len(ds_test), shuffle=True)
    
    return dl_train, dl_test