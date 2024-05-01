import torch
import torchvision
import os
import glob
import time 
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

from models import UNet, PretrainedUNet
from metrics import jaccard, dice

class Config:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 16





def main():
    data_path = r''
    
    pass

if __name__ == "__main__":
    main()