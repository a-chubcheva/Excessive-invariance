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



class Autoencoder(nn.Module):
  def __init__(self, input_size=28*28):
    super(Autoencoder, self).__init__()
    self.encoder = nn.Sequential(
        nn.Linear(input_size, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 64),
        nn.LeakyReLU())
    self.decoder = nn.Sequential(
        nn.Linear(64, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, input_size),   
        nn.Sigmoid())
  
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x


#-------------------------------------------------

class Classifier(nn.Module):
    def __init__(self, input_size=10):
      super(Classifier, self).__init__()
      self.classifier = nn.Sequential(
          nn.Linear(10,32),
          nn.ReLU(),
          nn.Linear(32,10))
    
    def forward(self, x):
      # predict by semantic logits - size 10
      x = self.classifier(x)
      return x
      
    def predict(self, semantic):
        output = self.classifier(semantic)
        _, preds = torch.max(output, 1)
        return preds.item()
        
    def predict_by_image(self, img, autoencoder ,is_vae=False):
        # input is (28, 28) or (1, 28, 28) tensor
        img = torch.reshape(img, (1, 28*28))
        if is_vae:
            _, logits, _ = autoencoder.encoder(img)
            semantic = logits[:,:10]
        else:
            semantic = autoencoder.encoder(img)[:,:10]
        return self.predict(semantic)
    
    


#-------------------------------------------------

class Reversal_layer(Function):
  @staticmethod
  def forward(self, x):
    return x
  
  @staticmethod
  def backward(self, grad):
    return grad.neg()
    
    
    
class Full_model(nn.Module):
  def __init__(self):
      super(Full_model, self).__init__()
      self.autoencoder = Autoencoder()
      self.is_vae = False

      self.semantic_classifier = Classifier()

      self.nuisance_classifier = nn.Sequential(
          nn.Linear(54, 32),
          nn.ReLU(),
          nn.Linear(32, 10))
      
  def forward(self, img):
      if self.is_vae:
          #logits, _, _ = self.autoencoder.encoder(img)
          _, logits, _ = self.autoencoder.encoder(img)
      else:
          logits = self.autoencoder.encoder(img)
      semantic = logits[:, :10]
      nuisance = logits[:, 10:]
      nuisance = Reversal_layer.apply(nuisance)
      class_output = self.semantic_classifier(semantic)
      domain_output = self.nuisance_classifier(nuisance)

      return class_output, domain_output
      
#-------------------------------------------------

class VAE(nn.Module):
  def __init__(self, input_size=28*28):
    super(VAE, self).__init__()
    self.hidden = nn.Sequential(
        nn.Linear(input_size, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 64),
        nn.LeakyReLU())

    self.mean = nn.Linear(64, 64)
    self.logvar = nn.Linear(64, 64)

    self.decoder = nn.Sequential(
        nn.Linear(64, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, input_size),   
        nn.Sigmoid())
    
  def encoder(self, x):
    hidden = self.hidden(x)
    mean = self.mean(hidden)
    logvar = self.logvar(hidden)
    
    #random shift
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mean + eps * std, mean, logvar

  def forward(self, x):
    x, mean, logvar = self.encoder(x)
    x = self.decoder(x)
    return x, mean, logvar


# ---------------------------------------------
class quasi_VAE(nn.Module):
  def __init__(self, input_size=28*28):
    super(quasi_VAE, self).__init__()
    self.hidden = nn.Sequential(
        nn.Linear(input_size, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 64),
        nn.LeakyReLU())

    self.decoder = nn.Sequential(
        nn.Linear(64, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, input_size),   
        nn.Sigmoid())
    
  def encoder(self, x, eps=0.1):
    mean = self.hidden(x)
    
    #random shift
    std = torch.randn_like(mean)
    return mean + eps * std

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x