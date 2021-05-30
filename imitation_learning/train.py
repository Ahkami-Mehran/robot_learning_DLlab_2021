from __future__ import print_function

import sys
sys.path.append("../") 

import glob
import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from config import Config
import torch

from utils import *
from agent.bc_agent import BCAgent
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm


def read_labels_generator(datasets_dir='../data'):
    file_names = glob.glob(os.path.join(datasets_dir, "*.gzip"))
    for data_file in file_names:
        y = []
        f = gzip.open(data_file, 'rb')
        data = pickle.load(f)
        y.extend(data["action"])
        y = np.array(y).astype("float32")
        yield y


def read_data_generator(datasets_dir="../data", frac=0.1, is_fcn=True):
    print(f'\nreading  data from {datasets_dir}')
    file_names = glob.glob(os.path.join(datasets_dir, "*.gzip"))
    print(f'files: {file_names}')

    for data_file in file_names:
        print(f'\n--- current file: {data_file} ---')
        X = []
        y = []
        f = gzip.open(data_file, 'rb')
        data = pickle.load(f)
        n_samples = len(data["state"])
        # Hint: to access images use state_img here!
        if is_fcn:
            X.extend(data["state"])
        else:
            X.extend(data["state_img"])

        y.extend(data["action"])

        X = np.array(X).astype("float32")
        y = np.array(y).astype("float32")

        # split data into training and validation set
        X_train, y_train = (
            X[:int((1 - frac) * n_samples)],
            y[:int((1 - frac) * n_samples)],
        )
        X_valid, y_valid = (
            X[int((1 - frac) * n_samples):],
            y[int((1 - frac) * n_samples):],
        )
        yield X_train, y_train, X_valid, y_valid

# def read_data(datasets_dir="../data", frac = 0.1):
#     """
#     This method reads the states and actions recorded in drive_manually.py 
#     and splits it into training/ validation set.
#     """
#     print("... read data")
#     data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
#     f = gzip.open(data_file,'rb')
#     data = pickle.load(f)

#     # get images as features and actions as targets
#     X = np.array(data["state"]).astype('float32')
#     y = np.array(data["action"]).astype('float32')

#     # split data into training and validation set
#     n_samples = len(data["state"])
#     X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
#     X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
#     return X_train, y_train, X_valid, y_valid

def skip_frames(input_data, input_label, skip_no, history_length):
    assert skip_no > 0, f'config.skip_n: {skip_no}. Must be > 0'
    N_train = input_data.shape[0]
    skipped_data = []
    skipped_label = []
    skip_pointer = history_length
    while skip_pointer < N_train:
        data = np.dstack(input_data[skip_pointer -
                                    history_length:skip_pointer + 1, ...])
        label = input_label[skip_pointer, ...]
        skipped_data.append(data)
        skipped_label.append(label)
        skip_pointer += skip_no
    skipped_data = np.array(skipped_data)
    skipped_label = np.array(skipped_label)
    return skipped_data, skipped_label

def preprocessing(X_train, y_train, X_valid, y_valid, conf):
    # --- preprocessing for state vector ---
    if conf.is_fcn:
        X_train, y_train = skip_frames(input_data=X_train,
                                       input_label=y_train,
                                       skip_no=conf.skip_frames,
                                       history_length=0)
        X_valid, y_valid = skip_frames(input_data=X_valid,
                                       input_label=y_valid,
                                       skip_no=conf.skip_frames,
                                       history_length=0)
        # X_train = X_train.squeeze()
        # X_valid = X_valid.squeeze()

        X_train = rgb2gray(X_train)
        X_valid = rgb2gray(X_valid)

        X_valid = X_valid.reshape([-1, 96, 96, 1])
        X_train = X_train.reshape([-1, 96, 96, 1])
        y_train = np.apply_along_axis(action_to_id, 1, y_train)
        y_valid = np.apply_along_axis(action_to_id, 1, y_valid)
        return X_train, y_train, X_valid, y_valid

    # --- preprocessing for image data ---
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (100, 150, 1)
    # X_train shape: (N_sample, H, W, 3)
    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)
    X_valid = X_valid.reshape([-1, 96, 96, 1])
    X_train = X_train.reshape([-1, 96, 96, 1])

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (200, 300, 1). Later, add a history of the last N images to your state so that a state has shape (200, 300, N).

    # Hint: you can also implement frame skipping
    # skip_frames: parameter similar to stride in CNN
    skip_n = conf.skip_frames
    # history_length: images representing the history in each data point.
    hist_len = conf.history_length
    # X_train shape: (2250, 200, 300, 3)
    X_train, y_train = skip_frames(input_data=X_train,
                                   input_label=y_train,
                                   skip_no=skip_n,
                                   history_length=hist_len)

    X_valid, y_valid = skip_frames(input_data=X_valid,
                                   input_label=y_valid,
                                   skip_no=skip_n,
                                   history_length=hist_len)

    y_train = np.apply_along_axis(action_to_id, 1, y_train)
    y_valid = np.apply_along_axis(action_to_id, 1, y_valid)

    return X_train, y_train, X_valid, y_valid


# def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

#     # TODO: preprocess your data here.
#     # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
#     # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
#     #    using action_to_id() from utils.py.
#     if not torch.cuda.is_available():
#         train_index = np.random.choice(X_train.shape[0], config.dev_size, replace = False)
#         X_train_sampled = rgb2gray(X_train[train_index])
#         X_train_sampled = X_train_sampled.reshape([-1, 96, 96, 1])

#         X_valid = rgb2gray(X_valid)
#         X_valid = X_valid.reshape([-1, 96, 96, 1])

#         y_train = y_train[train_index]
#         y_train_discret = np.apply_along_axis(action_to_id, 1, y_train)
#         y_valid = np.apply_along_axis(action_to_id, 1, y_valid)
#         return X_train_sampled, y_train_discret, X_valid, y_valid

#     X_train = rgb2gray(X_train)
#     X_valid = rgb2gray(X_valid)

#     y_train = np.apply_along_axis(action_to_id, 1, y_train)
#     y_valid = np.apply_along_axis(action_to_id, 1, y_valid)

#     # History:
#     # At first you should only use the current image as input to your network to learn the next action. Then the input states
#     # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    
#     return X_train, y_train, X_valid, y_valid


def calculate_weights(datasets_dir):
    """
    function for calculating weights for each class in unbalanced datasets
    """
    data_generator = read_labels_generator(datasets_dir=datasets_dir)
    weight_list = []
    for labels in data_generator:
        classes = np.unique(labels)
        counts = np.zeros_like(classes)
        for i, c in enumerate(classes):
            counts[i] = np.count_nonzero(labels == c)
        weights = 1.0 / counts
        weights = weights / np.linalg.norm(weights, ord=1)  # normalization
        side_engine_weights = (weights[1] +
                               weights[3]) / 2  # same weights for side engines
        weights[1] = weights[3] = side_engine_weights
        weight_list.append(weights)
    return torch.Tensor(weight_list).mean(dim=0).view(-1, 1)


def calculate_n_minibatches(datasets_dir, config):
    filenames = glob.glob(os.path.join(datasets_dir, "*.gzip"))
    f = gzip.open(filenames[0], 'rb')
    data = pickle.load(f)
    y = data["action"]
    data_per_file = (len(y) *
                     (1.0 - config.validation_frac)) / config.skip_frames
    n_minibatches_per_epoch = (len(filenames) *
                               data_per_file) / config.batch_size
    return int(n_minibatches_per_epoch)


def create_data_loader(X, y, batch_size, shuffle=True):
    """
    Creates pytorch dataloaders from numpy data and labels
    """
    tensor_X = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=2)
    return loader


def train_model(datasets_dir, config, model_dir="./models", tensorboard_dir="./tensorboard"):
    tb_saving_res = 3
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M")
    modelfilename = os.path.join(model_dir, f'agent_{timestamp}.pt')
    writer = SummaryWriter(log_dir=f'{tensorboard_dir}/experiment_{timestamp}')
    if not os.path.exists(model_dir): os.mkdir(model_dir)

    # --- write configuration ---
    conf_dict = {
        'skip': config.skip_frames,
        'history': config.history_length,
        'lr': config.learning_rate,
        'batch_size': config.batch_size,
        'n_epochs': config.n_epochs,
        'is_fcn': config.is_fcn
    }
    writer.add_text(tag='configuration', text_string=str(conf_dict))

    # --- agent setup ---
    class_weights = calculate_weights(datasets_dir=datasets_dir)
    print(f'Class weights: {class_weights.squeeze()}')
    agent = BCAgent(config, weights=None)
    agent.train_mode()
    agent.to_device()

    # --- add model summary to tensorboard ---
    writer.add_text(tag='model', text_string=str(agent.agent.net))

    # --- training loop ---
    training_loss = 0
    step = 0
    n_minibatches = calculate_n_minibatches(datasets_dir, config)
    for epoch in range(config.n_epochs):
        print(f'\n=== Epoch: {epoch + 1} ===')
        data_file_generator = read_data_generator(datasets_dir=datasets_dir,
                                                  frac=config.validation_frac,
                                                  is_fcn=config.is_fcn)

        for raw_data in data_file_generator:
            X_train, y_train, X_valid, y_valid = raw_data
            # --- preprocessing the raw data ---
            X_train, y_train, X_valid, y_valid = preprocessing(
                X_train, y_train, X_valid, y_valid, config)
            # --- data loader to get batches of data ---
            trainloader = create_data_loader(X_train,
                                             y_train,
                                             batch_size=config.batch_size)
            # --- batch training ---
            for (X, y) in tqdm(trainloader, desc='Batch training: '):
                loss = agent.update(X, y)
                training_loss += loss.item()
                step += 1
                # --- logging ---
                if (step + 1) % tb_saving_res == 0:
                    # --- record training loss ---
                    writer.add_scalar(tag='training_loss',
                                      scalar_value=(training_loss /
                                                    tb_saving_res),
                                      global_step=step)
                    training_loss = 0.0
                    # --- record validation loss ---
                    with torch.set_grad_enabled(False):
                        agent.test_mode()
                        y_pred = agent.predict(X=X_valid).detach()
                        val_loss = agent.prediction_loss(y_pred=y_pred,
                                                         y=y_valid)
                        # new_lr = agent.scheduler_step(val_loss) # ReduceLROnPlateau
                        new_lr = agent.scheduler_step(
                            (step + 1) /
                            n_minibatches)  # CosineAnnealingWarmRestarts
                        writer.add_scalar(tag='validation_loss',
                                          scalar_value=val_loss,
                                          global_step=step)
                        writer.add_scalar(tag='Learning_rate',
                                          scalar_value=new_lr,
                                          global_step=step)
                        agent.train_mode()

    # --- closing and saving ---
    writer.close()
    agent.save(modelfilename)
    print(f'Model saved in file: {modelfilename}')


if __name__ == "__main__":
    conf = Config()
    datasets_dir = './data'
    train_model(datasets_dir=datasets_dir, config=conf)
