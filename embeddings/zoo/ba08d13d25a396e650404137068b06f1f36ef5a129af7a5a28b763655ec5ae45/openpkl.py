import pickle
import torch

model = torch.load("./model_0.pkl")

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor])
