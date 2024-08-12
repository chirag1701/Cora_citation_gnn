# -*- coding: utf-8 -*-
"""Cora_gnn.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hOrve_gayYGk3xAHzKVFzuQJmKUbbYeK
"""

import torch
import torchvision

!pip install torch torchvision torchaudio

!pip install torch_geometric

import torch_geometric

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
dataset = Planetoid(root = 'data/Planetoid', name = 'Cora', transform = NormalizeFeatures())

print(f"Number of graphs : {len(dataset)}")
print(f'Number of features : {dataset.num_features}')
print(f'Number of classes : {dataset.num_classes}')
print(100*"*")

data = dataset[0]
print(data)

print(f"Number of nodes : {data.num_nodes}")
print(f"Number of edges : {data.num_edges}")

print(f'Number of training nodes: {data.train_mask.sum()}')   # sum of all True values give us the total number of training nodes

print(f"Training node label rate : {(data.train_mask.sum()/data.num_nodes):.2f}")

print(f"Undirected data : {data.is_undirected()}")

data.x.shape  # (Number of nodes, features)

data.y

len(data.test_mask)==data.num_nodes

data.test_mask

data.edge_index.t() # individual node connection information

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
   def __init__(self, hidden_channels):
    super(GCN,self).__init__()
    torch.manual_seed(41)
    self.conv1 = GCNConv(dataset.num_features, hidden_channels)
    self.conv2 = GCNConv(hidden_channels, hidden_channels)
    self.conv3 = GCNConv(hidden_channels, hidden_channels)
    self.out = Linear(hidden_channels, dataset.num_classes)
   def forward(self,x, edge_index) :  # no batch_index needed since we have only 1 graph
    x = self.conv1( x ,edge_index)
    x = x.relu()
    x = self.conv2( x , edge_index)
    x = x.relu()
    x = self.conv3( x, edge_index)
    x = x.relu()  # Introduces non linearity at each step
    x = F.dropout(x, p = 0.4, training = self.training)
    x = self.out(x) # applies the final linear layer
    x = F.softmax(x, dim = 1)
    return x
model = GCN(hidden_channels = 32)
print(model)

model = GCN(hidden_channels = 32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)
learning_rate = 0.01
decay = 5e-5
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = decay)
criterion_loss = torch.nn.CrossEntropyLoss()

def train():
  model.train()
  optimizer.zero_grad()
  output = model(data.x, data.edge_index)
  loss = criterion_loss(output[data.train_mask], data.y[data.train_mask])
  loss.backward()
  optimizer.step()
  return loss

def test():
  model.eval()
  output = model(data.x, data.edge_index)
  pred = output.argmax(dim = 1)
  test_correct = pred[data.test_mask] == data.y[data.test_mask] # it will be an array
  test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
  return test_acc

losses = []
for epoch in range(0, 1001):
    loss = train()
    losses.append(loss)
    if epoch % 100 == 0:
      print(f'Epoch: {epoch}, Loss: {loss:.4f}')

import seaborn as sns
losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
loss_indices = [i for i,l in enumerate(losses_float)]
sns.lineplot(x = loss_indices, y= losses_float)

test_acc = test()
print(test_acc)

"""#**VISUALISATION**"""

!pip install moviepy

# prompt: write code to make gif of epochs tsne using moviepy

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import numpy as np

def visualize(epoch, loss):
  model.eval()
  z = model(data.x, data.edge_index)
  z = TSNE(n_components=2).fit_transform(z.cpu().detach().numpy())
  plt.figure(figsize=(10, 10))
  plt.scatter(z[:, 0], z[:, 1], c=data.y.cpu(), cmap="tab10")
  plt.title(f"Epoch: {epoch}, Loss: {loss:.4f}")
  plt.savefig(f"epoch_{epoch}.png")
  plt.close()
frames = []
for epoch in range(0, 101):
    loss = train()
    if epoch % 10 == 0:
      visualize(epoch, loss)
      frames.append(f"epoch_{epoch}.png")

clip = ImageSequenceClip(frames, fps=2)
clip.write_gif("tsne_animation.gif", fps=2)
clip

# prompt: write code to make gif of epochs tsne using moviepy

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import numpy as np

def visualize(epoch, loss):
  model.eval()
  z = model(data.x, data.edge_index)
  z = TSNE(n_components=2).fit_transform(z.cpu().detach().numpy())
  plt.figure(figsize=(10, 10))
  plt.scatter(z[:, 0], z[:, 1], c=data.y.cpu(), cmap="tab10")
  plt.title(f"Epoch: {epoch}, Loss: {loss:.4f}")
  plt.savefig(f"epoch_{epoch}.png")
  plt.close()
frames = []
for epoch in range(0, 101):
    loss = train()
    if epoch % 10 == 0:
      visualize(epoch, loss)
      frames.append(f"epoch_{epoch}.png")

clip = ImageSequenceClip(frames, fps=2)
clip.write_gif("tsne_animation.gif", fps=2)
clip

# prompt: write code to make gif of epochs tsne using moviepy

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import numpy as np

def visualize(epoch, loss):
  model.eval()
  z = model(data.x, data.edge_index)
  z = TSNE(n_components=2).fit_transform(z.cpu().detach().numpy())
  plt.figure(figsize=(10, 10))
  plt.scatter(z[:, 0], z[:, 1], c=data.y.cpu(), cmap="tab10")
  plt.title(f"Epoch: {epoch}, Loss: {loss:.4f}")
  plt.savefig(f"epoch_{epoch}.png")
  plt.close()
frames = []
for epoch in range(0, 1001):
    loss = train()
    if epoch % 100 == 0:
      visualize(epoch, loss)
      frames.append(f"epoch_{epoch}.png")

clip = ImageSequenceClip(frames, fps=3)
clip.write_gif("tsne_animation.gif", fps=3)
clip

!ls -l tsne_animation.gif

from google.colab import files
files.download('tsne_animation.gif')

from IPython.display import Image
Image(filename='tsne_animation.gif')
