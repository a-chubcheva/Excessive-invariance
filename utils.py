import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchsummary
from torch.autograd import Variable, Function
from copy import deepcopy
from itertools import chain
import math
import pandas as pd
from PIL import Image




def train_autoencoder(model, criterion, optimizer, dataloader, num_epochs):
  model.train()
  best_model_wts = deepcopy(model.state_dict())
  best_loss = 10000.
  loss_history = []
  for epoch in range(num_epochs):
    running_loss = 0
    for inputs, _ in dataloader:
        inputs = inputs.view(inputs.shape[0], -1)
        inputs = Variable(inputs.to(device))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss = running_loss / len(dataloader.dataset)
    loss_history.append(epoch_loss)
    if epoch_loss < best_loss:
        best_model_wts = deepcopy(model.state_dict())
    print('epoch [{}/{}], loss:{:.4f}\n'.format(epoch+1, num_epochs, loss.item()))

  model.load_state_dict(best_model_wts)
  return model, loss_history
  


def train_classifier(classifier, autoencoder, criterion, optimizer, dataloaders, dataset_sizes, num_epochs):

  for param in autoencoder.parameters():
    param.requires_grad = False

  best_model_wts = deepcopy(classifier.state_dict())
  best_acc = 0.0

  loss_history = {'train': [], 'val': []}
  acc_history = {'train': [], 'val': []}

  for epoch in range(num_epochs):
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()
      
      running_loss = 0.0
      running_correct = 0

      for inputs, labels in dataloaders[phase]:
        inputs = inputs.view(inputs.shape[0], -1)
        inputs = Variable(inputs.to(device))
        labels = Variable(labels.to(device))
        optimizer.zero_grad()

        logits = autoencoder.encoder(inputs)
        semantic = logits[:, :10]

        outputs = classifier(semantic)
        _, preds = torch.max(outputs, 1)  
        loss = criterion(outputs, labels)
        if phase == 'train':
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * inputs.size(0)
        running_correct += torch.sum(labels == preds.data)

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_correct.double() / dataset_sizes[phase]
      loss_history[phase].append(epoch_loss)
      acc_history[phase].append(epoch_acc)

      if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(classifier.state_dict())

      print('{} epoch [{}/{}], loss: {:.4f}, acc: {:.4f}'.format(
                phase, epoch+1, num_epochs, epoch_loss, epoch_acc))
    print()

  classifier.load_state_dict(best_model_wts)
  return classifier, loss_history, acc_history
  


      
        
  
def attack(model, input, target, num_iter, alpha):
  model.eval()
  with torch.set_grad_enabled(True):
    img = deepcopy(input)
    output_targ = model(target)[:, :10]
    best_loss = 10000.
    best_img = deepcopy(img)
    img = Variable(img, requires_grad=True)
    for i in range(num_iter):
      
      output = model(img)[:, :10]
      loss = torch.norm(output - output_targ, p=2)
      loss.backward(retain_graph=True)

      if loss.item() < best_loss:
        best_loss = loss.item()
        best_img = deepcopy(img)
      
      #step
      img.data -= alpha * torch.sign(img.grad.data)

  return best_img, best_loss
  

