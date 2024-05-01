import numpy
import h5py
import torch
from pathlib import Path
import torch.utils.data
import torch.utils.data.dataset
from utils import Config
import torchvision.transforms as transforms


def create_dataloader(config:Config):
    train_set = H5Dataset(config, 'train')
    valid_set = H5Dataset(config, 'valid')

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    return {
        'train': train_loader,
        'valid': valid_loader,
    }

class H5Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, config:Config, selection):
        # initial check
        assert selection in ["train", "valid"]
        dataset_root = Path(config.dataset_path, selection)

        # load volume
        self.volume = [] 
        for file_path in dataset_root.rglob('*.h5'):
            print(f'loading {file_path}')
            image = h5py.File(file_path)['image'][:]
            mask = h5py.File(file_path)['mask'][:]
            self.volume.extend = [file_path, image, mask]

        # Build a table to convert global indices into patient-specific indices.
        num_slice = [v.shape[0] for p, v, m in self.volume]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slice[i] for i in range(len(num_slice))], []),
                sum([list(range(x)) for x in num_slice], []),
            )
        )

        # Image transform
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        path, volume, mask = self.volumes[patient]
        image = volume[slice_n]
        mask = mask[slice_n]

        return image, mask, path

# test funcion
if __name__ == '__main__':
    config = Config
    config.dataset_path = r'dataset\h5'
    dataset = H5Dataset(config, 'train')
