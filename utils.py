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
    
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
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
  


def train_classifier(classifier, autoencoder, criterion, optimizer, dataloaders, dataset_sizes, num_epochs, is_vae=False):
    
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  for param in autoencoder.parameters():
    param.requires_grad = False

  best_model_wts = deepcopy(classifier.state_dict())
  best_acc = 0.0

  loss_history = {'train': [], 'val': []}
  acc_history = {'train': [], 'val': []}

  for epoch in range(num_epochs):
    for phase in ['train', 'val']:
      if phase == 'train':
        classifier.train()
      else:
        classifier.eval()
      
      running_loss = 0.0
      running_correct = 0

      for inputs, labels in dataloaders[phase]:
        inputs = inputs.view(inputs.shape[0], -1)
        inputs = Variable(inputs.to(device))
        labels = Variable(labels.to(device))
        optimizer.zero_grad()

        if is_vae:
            logits, _, _ = autoencoder.encoder(inputs)
        else: 
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
  


def train_full_model(model, optimizer, class_criterion, domain_criterion, dataloaders, num_epochs, is_vae=False):
    
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  loss_history = {'train': [], 'val': []}
  acc_history = {'train': [], 'val': []}
  acc_history_nuisance = {'train': [], 'val': []}

  for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        epoch_class_loss = 0.
        epoch_domain_loss = 0.
        running_correct = 0
        running_correct_nuisance = 0
        running_loss = 0.

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.view(inputs.shape[0], -1)
            inputs = Variable(inputs.to(device))
            class_labels = Variable(labels.to(device))

            #make domain binary labels
            domain_labels = torch.zeros((len(inputs), 10)).long()
            for i in range(len(labels.data)):
                domain_labels.data[i][labels.data[i]] = 1
            domain_labels = Variable(domain_labels.to(device))

            optimizer.zero_grad()

            class_output, domain_output = model(inputs)
            _, class_preds = torch.max(class_output, 1)
            class_loss = class_criterion(class_output, class_labels)
            running_correct += torch.sum(class_labels == class_preds.data)          
        
            domain_loss = domain_labels * torch.log(domain_output)
            where_nan = np.isnan(domain_loss.data)
            domain_loss.data[where_nan] = 0
            domain_loss = torch.sum(domain_loss)
            
            #check nuisance classifier
            _, nuisance_preds = torch.max(domain_output, 1)
            running_correct_nuisance += torch.sum(class_labels == nuisance_preds.data)  

          #  epoch_class_loss += class_loss
          #  epoch_domain_loss += domain_loss
            loss = class_loss - domain_loss
            running_loss += loss * inputs.size(0)
            if phase == 'train':
                loss.backward()
                optimizer.step()

        epoch_acc = running_correct.double() / len(dataloaders[phase].dataset)
        epoch_acc_nuisance = running_correct_nuisance.double() / len(dataloaders[phase].dataset)
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        acc_history[phase].append(epoch_acc)
        acc_history_nuisance[phase].append(epoch_acc_nuisance)
        loss_history[phase].append(epoch_loss)
            
        print('{} epoch [{}/{}], class acc: {:.4f}, loss: {:.4f}'.format(
                  phase, epoch+1, num_epochs, epoch_acc, epoch_loss))
    
  return model, acc_history, loss_history, acc_history_nuisance

        
        
  
def attack(autoencoder, classifier, full_model, input, target, num_iter, alpha, df=None):
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  with torch.set_grad_enabled(True):
    input = torch.reshape(input, (1, 28*28)).to(device)
    target = torch.reshape(target, (1, 28*28)).to(device)

    first_flip = None 
    first_flip_da = None   # da = domain adaptation
    init_label = classifier.predict_by_image(input, autoencoder)
    target_label = classifier.predict_by_image(target, autoencoder)
    output_targ = autoencoder.encoder(target)[:, :10]
    best_loss = 10000.

    img = deepcopy(input)
    best_img = deepcopy(img)
    img = Variable(img, requires_grad=True).to(device)

    for i in range(num_iter):
      output = autoencoder.encoder(img)[:, :10]
      loss = torch.norm(output - output_targ, p=2)
      loss.backward(retain_graph=True)

      if loss.item() < best_loss:
        best_loss = loss.item()
        best_img = deepcopy(img)
      
      #gradien descent step
      img.data -= alpha * torch.sign(img.grad.data)

      curr_label = classifier.predict_by_image(img, autoencoder)
      curr_label_da = full_model.semantic_classifier.predict_by_image(img, autoencoder)
      
      if curr_label == target_label and first_flip is None:
        first_flip = i
      if curr_label_da == target_label and first_flip_da is None:
        first_flip_da = i
      '''
      if curr_label != init_label and first_flip is None:
        first_flip = i
      if curr_label_da != init_label and first_flip_da is None:
        first_flip_da = i
      '''

  if df is not None:
      df = df.append({'input': init_label,
                      'target': target_label,
                      'flip': first_flip,
                      'flip_da': first_flip_da},
                      ignore_index=True)

  return best_img, best_loss, df
  

def to_jpg_and_back(img):
  img = torch.reshape(img, (28, 28))
  img = img.data.cpu().numpy()

  #make jpg
  img = img * 255.0
  img = np.clip(img, 0, 255).astype(np.uint8)
  img_pil = torchvision.transforms.ToPILImage()(img)
  img_pil.save('img.jpg', 'JPEG')

  #back to tensor
  img_back = (np.asarray(Image.open('img.jpg')) / 255.0).astype(np.float32)
  transform = transforms.ToTensor()
  img_back = transform(img_back)
  img_back = torch.reshape(img_back, (1, 28, 28))
  
  return img_back