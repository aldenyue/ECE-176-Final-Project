import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.Models import ContextInpainter, AdversarialDiscriminator


#Save ContextEncoder and AdversarialDiscriminator Weights
def save_weights(theModel, theAD, path= "./weights/"):
    torch.save(theModel.state_dict(), path + "ContextInpainter")
    torch.save(theAD.state_dict(), path + "AdversarialDiscriminator")

#Load Our model
def load_weights(path= "./weights/"):
    model = ContextInpainter()
    AD = AdversarialDiscriminator()
    
    model.load_state_dict(torch.load(path + "ContextInpainter"))
    AD.load_state_dict(torch.load(path + "AdversarialDiscriminator"))
    
    model.eval()
    AD.eval()
    return model, AD