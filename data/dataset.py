from torch.utils.data import Dataset
import torch

class RUL_Dataset(Dataset):
    def __init__(self, x_data, y_data):
        super(RUL_Dataset, self).__init__()
        self.x_data = x_data
        self.y_data = y_data
    
    
    def __len__(self):
        return len(self.x_data)
    
    
    def __getitem__(self, index):
        x = torch.tensor(self.x_data[index]).float()
        y = torch.tensor(self.y_data[index]).float()
        if len(y.size()) != 0:
            y = y[0]
        return x, y
