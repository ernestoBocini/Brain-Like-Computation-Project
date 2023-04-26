import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import explained_variance_score
from tqdm import tqdm

from utils import *

path_to_data = 'data'



model_path = "data/resnet50_improved/model_resnet50_imp.pth"

# uncomment the following if also interested in loading all the training and the validation datasets
stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val, spikes_test = load_it_data(path_to_data)



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
        
# define the model
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
# Replace final fully connected layer with a new layer for your task
num_neurons = 168  # number of output neurons
model.fc = nn.Linear(model.fc.in_features, num_neurons)

# Define PyTorch data loaders for training and validation datasets
batch_size = 64
val_dataset = MyDataset(stimulus_test, spikes_test)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define optimizer and loss function for training
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

model.load_state_dict(torch.load(model_path))


# Evaluate on test

model.eval()
with torch.no_grad():
    val_preds = []
    for data, target in val_loader:
        output = model(data)
        test_preds.append(output.numpy())
    val_preds = np.concatenate(val_preds, axis=0)
    ev = explained_variance_score(spikes_test, test_preds)
    corr = compute_corr(spikes_test, test_preds)

print('The explained variance is:', ev)
print('The correlation between predicted values and actual values is:', corr)