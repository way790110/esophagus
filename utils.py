import torch
class Config:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = None
        self.batch_size = None
        self.model = None
        self.optimizer = None