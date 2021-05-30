from __future__ import print_function

import sys
sys.path.append("../") 

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

config = Config()

def read_data(datasets_dir="../data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.
    if not torch.cuda.is_available():
        train_index = np.random.choice(X_train.shape[0], config.dev_size, replace = False)
        X_train_sampled = rgb2gray(X_train[train_index])
        X_train_sampled = X_train_sampled.reshape([-1, 96, 96, 1])

        X_valid = rgb2gray(X_valid)
        X_valid = X_valid.reshape([-1, 96, 96, 1])

        y_train = y_train[train_index]
        y_train_discret = np.apply_along_axis(action_to_id, 1, y_train)
        y_valid = np.apply_along_axis(action_to_id, 1, y_valid)
        return X_train_sampled, y_train_discret, X_valid, y_valid

    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)

    y_train = np.apply_along_axis(action_to_id, 1, y_train)
    y_valid = np.apply_along_axis(action_to_id, 1, y_valid)

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    
    return X_train, y_train, X_valid, y_valid


def calculate_n_minibatches(data_size):
    n_minibatches_per_epoch = (data_size / batch_size)
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


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M")
    modelfilename = os.path.join(model_dir, f'agent_{timestamp}.pt')
    writer = SummaryWriter(log_dir=f'{tensorboard_dir}/experiment_{timestamp}')

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
     # --- write configuration ---
    conf_dict = {
        'dev_size': config.dev_size,
        # 'skip': config.skip_frames,
        # 'history': config.history_length,
        # 'lr': config.learning_rate,
        # 'batch_size': config.batch_size,
        # 'n_epochs': config.n_epochs,
        # 'is_fcn': config.is_fcn
    }
    writer.add_text(tag='configuration', text_string=str(conf_dict))

    print("... train model")


    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    agent = BCAgent(config, weights=None)
    agent.train_mode()
    agent.to_device()

    writer.add_text(tag='model', text_string=str(agent.agent.net))
    
    # tensorboard_eval = Evaluation(tensorboard_dir)
    tb_saving_res = 3
    training_loss = 0
    step = 0
    for epoch in range(config.epochs):
        print(f'\n=== Epoch: {epoch + 1} ===')

        trainloader = create_data_loader(X_train, y_train, batch_size=batch_size)
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

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # 
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     for i % 10 == 0:
    #         # compute training/ validation accuracy and write it to tensorboard
    #         tensorboard_eval.write_episode_data(...)
      
    # TODO: save your agent
    # model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    # print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":
    
    # read data    
    X_train, y_train, X_valid, y_valid = read_data("../data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=1000, batch_size=64, lr=1e-4)
 
