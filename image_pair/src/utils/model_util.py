"""Model utility functions."""

import torch
import os.path
import numpy as np
import matplotlib.pyplot as plt
from src.utils.logging_util import LoggingUtils

logger = LoggingUtils.configure_logger(log_name=__name__)



def save_model(model, filepath="model.pth"):
    """Funtion to save model weights to a file."""
    
    save_dir = os.path.dirname(filepath)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.state_dict(), filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(model_class, filepath, device, model_args):
    """Function to load model weights and initialize the model."""
    
    model = model_class(**model_args).to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    logger.info(f"Model loaded from {filepath}")
    return model


def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('Number of parameters: {} ({:.2f} million)'.format(nb_param, nb_param/1e6))
