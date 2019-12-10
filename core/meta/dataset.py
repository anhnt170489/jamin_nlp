from torch.utils.data import (
    Dataset
)


class JaminDataset(Dataset):
    """Dataset instance for training/evaluating"""

    def __init__(
            self, data
    ):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
