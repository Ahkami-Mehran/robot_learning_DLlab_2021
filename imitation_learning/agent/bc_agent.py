from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from agent.networks import CNN
from utils import DEVICE, tt, _x, _y
# from agent.resnet18 import Resnet18

class BaseBCAgent(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update(self, X_batch, y_batch):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def test_mode(self):
        self.net.eval()

    def train_mode(self):
        self.net.train()

    def prediction_loss(self, y_pred, y):
        loss = self.criterion(y_pred, _y(y))
        return loss

    def to_device(self):
        self.net.to(DEVICE)

    def load(self, file_name, to_cpu=False):
        state = torch.load(file_name, map_location=DEVICE)
        self.net.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        # pytorch does not have optimizer.to(DEVICE) so ...
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(DEVICE)

    def save(self, file_name):
        state = {
            'model_state': self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }
        torch.save(state, file_name)


class BCAgentCNN(BaseBCAgent):
    def __init__(self, history_length, learning_rate, weights_classes):
        weights_classes = None if weights_classes is None else weights_classes.to(
            DEVICE)
        self.net = CNN(history_length=history_length, n_classes=4)
        # self.net = Resnet18(history_length=history_length, n_classes=4)
        self.criterion = nn.CrossEntropyLoss(weight=weights_classes)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(),
                                           lr=learning_rate)
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, threshold=0.00001)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1, T_mult=3)

    def update(self, X_batch, y_batch):
        self.optimizer.zero_grad()
        y_batch_pred = self.net(_x(X_batch))
        loss = self.criterion(y_batch_pred, _y(y_batch))
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, X):
        outputs = self.net(_x(X)).detach()
        return outputs


class BCAgent:
    def __init__(self, conf, weights=None):
        self.agent = BCAgentCNN(history_length=conf.history_length,
                                learning_rate=conf.learning_rate,
                                weights_classes=weights)

    def update(self, X_batch, y_batch):
        return self.agent.update(X_batch, y_batch)

    def predict(self, X):
        return self.agent.predict(X)

    def prediction_loss(self, y_pred, y):
        return self.agent.prediction_loss(y_pred, y)

    def test_mode(self):
        self.agent.test_mode()

    def train_mode(self):
        self.agent.train_mode()

    def to_device(self):
        self.agent.to_device()

    def load(self, file_name, to_cpu):
        self.agent.load(file_name, to_cpu=to_cpu)

    def save(self, file_name):
        self.agent.save(file_name)

    def scheduler_step(self, val):
        # self.agent.lr_scheduler.step(val) # ReduceLROnPlateau
        self.agent.lr_scheduler.step(val)  # CosineAnnealingWarmRestarts
        return self.agent.optimizer.param_groups[0]['lr']
