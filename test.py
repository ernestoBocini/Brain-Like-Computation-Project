import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import explained_variance_score
from tqdm import tqdm

from utils import *

stimulus_test
objects_test

path_to_data = 'data'

stimulus_test, spikes_test, objects_test = load_test_data(path_to_data)


class MyDataset(data.Dataset):
        def __init__(self, stimulus, spikes):
            self.stimulus = stimulus
            self.spikes = spikes

        def __getitem__(self, index):
            x = self.stimulus[index]
            y = self.spikes[index]
            return x, y

        def __len__(self):
            return len(self.stimulus)
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    # Replace final fully connected layer with a new layer for your task
    num_neurons = 168  # number of output neurons
    model.fc = nn.Linear(model.fc.in_features, num_neurons)

    # Define PyTorch data loaders for training and validation datasets
    batch_size = 64
    train_dataset = MyDataset(stimulus_train, spikes_train)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = MyDataset(stimulus_val, spikes_val)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define optimizer and loss function for training
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.load_state_dict(torch.load(model_path))
    



model.eval()
with torch.no_grad():
    test_preds = []
    for data, target in val_loader:
        output = model(data)
        val_preds.append(output.numpy())
    val_preds = np.concatenate(val_preds, axis=0)
    ev = explained_variance_score(spikes_test, test_preds)
    corr = compute_corr(spikes_test, test_preds)

print('The explained variance is:', ev)
print('The correlation between predicted values and actual values is:', corr)