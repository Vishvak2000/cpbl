import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


def calculate_layer_output_length(input_length, kernel_size, stride=1, padding=0, dilation=1):
    """Calculate the output length of a convolutional layer."""
    return ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

def calculate_required_input_length(desired_output_length, model_params):
    """
    Calculate the required input length to achieve a desired output length
    after all convolutions and cropping operations.
    
    Args:
        desired_output_length: The desired length of the final output
        model_params: dict containing:
            - conv1_kernel_size: Size of first convolution kernel
            - n_dil_layers: Number of dilated convolution layers
            - dilation_kernel_size: Kernel size for dilated convolutions
    
    Returns:
        required_input_length: The required input length
    """
    # Work backwards from the desired output length
    current_length = desired_output_length
    
    # First conv layer (no dilation)
    current_length = current_length + model_params['conv1_kernel_size'] - 1
    
    # Account for each dilated layer
    for i in range(model_params['n_dil_layers'], 0, -1):
        dilation = 2 ** i
        kernel_effect = (model_params['dilation_kernel_size'] - 1) * dilation
        current_length = current_length + kernel_effect

    return current_length + 2
