from torch.utils.data import Dataset

class CreateTweetsDataset(Dataset):
    def __init__(self,data) -> None:
        super().__init__()
        self.data = [[data[key][i] for key in data] for i in range(len(data['label']))]
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
        