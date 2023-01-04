# Imports
# -------

from   torch.utils.data import Dataset, DataLoader



# Graph Dataset Class
# -------------------

class GraphDataset(Dataset):

  def __init__(self, pd_data):
    # data loading
    self.z      = pd_data['z'].values
    self.y      = pd_data['y'].values
    self.status = pd_data['status'].values

    self.n_samples = self.z.shape[0]

  def __getitem__ (self,index):
    return self.z[index], self.y[index], self.status[index]
    # dataset[0]

  def __len__(self):
    # len(dataset)
    return self.n_samples


