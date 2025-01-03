import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from torch_geometric.utils import to_dense_batch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MelanomaDataset(Dataset):
    def __init__(self, path, full = False):
        """
        Args:
            path (String): Path to the pickle files
            full (Bool): Whether to use full data or sub-sampled data
        """
        if full:
            suffix = '_full'
        else:
            suffix = ''
        with open(os.path.join(path, 'pc'+suffix+'.pkl'), 'rb') as handle:
            self.subsampled_pcs = pickle.load(handle)
            self.subsampled_pcs = [torch.tensor(StandardScaler().fit_transform(self.subsampled_pcs[i]), dtype=torch.float).to(device) for i in range(len(self.subsampled_pcs))]
        with open(os.path.join(path, 'patient_list'+suffix+'.pkl'), 'rb') as handle:
            self.subsampled_patient_ids = pickle.load(handle)
        self.labels = np.load(os.path.join(path, 'labels'+suffix+'.npy'))
        self.labels = torch.LongTensor(self.labels)


    def __len__(self):
        """Returns the total number of samples."""
        return len(self.subsampled_pcs)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample

        Returns:
            dict: Dictionary containing the data and the label
        """
        sample = self.subsampled_pcs[idx]
        label = self.labels[idx]

        return sample, label

class MelanomaDataloader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        """
        Args:
            dataset (CustomDataset): The dataset to load.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data at the start of each epoch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))
        self.current_idx = 0
        self.sizes = torch.Tensor([i[0].shape[0] for i in self.dataset]).long().to(device)

        if self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        """Shuffles the dataset indices."""
        import random
        random.shuffle(self.indices)

    def __iter__(self):
        """Returns the iterator object itself."""
        self.current_idx = 0  # Reset at the start of each iteration
        if self.shuffle:
            self._shuffle_data()
        return self

    def __next__(self):
        """Fetches the next batch of data."""
        if self.current_idx >= len(self.dataset):
            raise StopIteration  # End of the dataset

        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[start_idx:end_idx]

        batch_data = [self.dataset[idx][0] for idx in batch_indices]
        batch_labels = torch.LongTensor([self.dataset[idx][1] for idx in batch_indices]).to(device)
        sizes = self.sizes[batch_indices]
        indices = torch.cumsum(sizes, dim=0).long()
        X = torch.cat([i for i in batch_data], dim=0)
        batches = torch.zeros(X.shape[0]).long().to(device)
        for i in range(1,len(batch_indices)):
            batches[indices[i-1].item():indices[i].item()] = i
        X, mask = to_dense_batch(X, batches)
        self.current_idx += self.batch_size
        
        return X, mask, batch_labels

# data = [[1, 2], [3, 4], [5, 6], [7, 8]]  # Example data
# labels = [0, 1, 0, 1]                    # Corresponding labels

# Instantiate the custom dataset
# custom_dataset = (data, labels)

# # Create a DataLoader with batching, shuffling, etc.
# custom_dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True)

# # Iterate through the DataLoader
# for batch in custom_dataloader:
#     print(batch)
