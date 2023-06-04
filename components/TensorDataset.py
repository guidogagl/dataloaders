import torch

# utility class to store supervised learning tabular data
# X (num_samples, n_features), y (num_samples, )

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        assert len( X ) == len( y ), "X, y should have the same "
        assert torch.is_tensor(X) and torch.is_tensor(y), "Both X and y should be tensors"
        
        self.n_samples_ = int(X.size(0).item())
        self.n_features_ = int(X.size(1).item())

        self.n_classes_ = int( len( torch.unique( y )))

        self.X_ = X.float()
        self.y_ = y.long() 

    def __getitem__(self, idx):
        return self.X_[idx], self.y_[idx]

    def __len__(self):
        return self.n_samples_