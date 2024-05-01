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

from model_zoo import * 
from metrics import jaccard, dice
from utils import Config


def main():
    config = Config()
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.epochs = 20
    config.batch_size = 16
    config.model = PretrainedUNet(
        in_channels=1,
        out_channels=2, 
        batch_norm=True, 
        upscale_mode="bilinear"
    )
    config.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.0005)
    data_path = Path('dataset', 'h5') 

    
    pass

if __name__ == "__main__":
    main()