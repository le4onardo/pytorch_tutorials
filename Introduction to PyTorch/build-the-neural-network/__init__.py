import os
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from NeuralNetwork import NeuralNetwork

device = (
  "cuda"
  if torch.cuda.is_available()
  else "mps"
  if torch.backends.mps.is_available()
  else "cpu"
)

print(f"Using {device} device")

model= NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)

logits = model(X)
print(logits)

pred_probab = nn.Softmax(dim=1)(logits)
print("pred prob", pred_probab)

y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


### MODEL PARAMETERS

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")