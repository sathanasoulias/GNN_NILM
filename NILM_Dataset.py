# Imports
# -------

import torch.utils.data as data_utils
from   torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from collections import defaultdict
import numpy as np
import torch


# NILM Dataset 
# ------------

class NILMDataset(data_utils.Dataset):
    def __init__(self, x, y, status, window_size=480, stride=30):
        self.x           = x
        self.y           = y
        self.status      = status
        self.window_size = window_size
        self.stride      = stride

    def __len__(self):
        return int(np.ceil((len(self.x) - self.window_size) / self.stride) + 1)

    def __getitem__(self,index):
        start_index = index * self.stride
        end_index   = np.min((len(self.x), index * self.stride + self.window_size))
        
        x        = self.padding_seqs(self.x[start_index: end_index])
        y        = self.padding_seqs(self.y[start_index: end_index])
        status   = self.padding_seqs(self.status[start_index: end_index])
        
        channels = self.y.shape[1]
        x        = torch.Tensor(x).view((1,-1))
        y        = torch.Tensor(y).view((channels,-1))
        status   = torch.Tensor(status).view((channels,-1))
        
        return x, y, status 
    
    def padding_seqs(self, in_array):
        if len(in_array) == self.window_size:
            return in_array
        try:
            out_array = np.zeros((self.window_size, in_array.shape[1]))
        except:
            out_array = np.zeros(self.window_size)

        length             = len(in_array)
        out_array[:length] = in_array
        return out_array

    
    

# UK_Dale Parser
# --------------

class UK_Dale_Parser:
    """
    Reading UKDale dataset data.
    Should contain:
    x (n,)              : aggregate signal
    y (n,num_appliances): individual device consumption signals
    """
    
    def __init__(self,args):
        self.data_location   = args.data_location
        self.house_indicies  = args.house_indicies
        self.appliance_names = args.appliance_names
        self.sampling        = args.sampling
        
        self.cutoff        =  [args.cutoff[appl]    for appl in ['aggregate']+args.appliance_names]
        self.threshold     =  [args.threshold[appl] for appl in args.appliance_names]
        self.min_on        =  [args.min_on[appl]    for appl in args.appliance_names]
        self.min_off       =  [args.min_off[appl]   for appl in args.appliance_names]

        self.val_size      =  args.validation_size
        self.window_size   =  args.window_size
        self.window_stride =  args.window_stride
        
        self.x, self.y     = self.load_data()
        if args.normalze == 'mean':
          print('normalize data')
          self.x_mean = np.mean(self.x)
          self.x_std  = np.std(self.x)
          self.x = (self.x - self.x_mean) / self.x_std  
        self.status        = self.compute_status(self.y)
    
    
    
    def load_data(self):
        
        for appliance in self.appliance_names:
        	assert appliance in ['dishwasher', 'fridge','microwave', 'washing_machine', 'kettle']
        
        for house_id in self.house_indicies:
            assert house_id in [1, 2, 3, 4, 5]
        
        directory = Path(self.data_location).joinpath('uk_dale')  
        
        for house_id in self.house_indicies:
            house_folder = directory.joinpath('house_' + str(house_id))
            house_label  = pd.read_csv(house_folder.joinpath('labels.dat'),    sep=' ', header=None)    
            house_data   = pd.read_csv(house_folder.joinpath('channel_1.dat'), sep=' ', header=None) #aggregate
            
            #read aggregate data and resample
            house_data.columns = ['time','aggregate']
            house_data['time'] = pd.to_datetime(house_data['time'], unit = 's')
            house_data         = house_data.set_index('time').resample(self.sampling).mean().fillna(method='ffill', limit=30)
            
            appliance_list = house_label.iloc[:, 1].values
            app_index_dict = defaultdict(list)
            
            #find if device exists in house and create a dictionary that contains the channel names
            for appliance in self.appliance_names:
                try:
                    idx = appliance_list.tolist().index(appliance)
                    app_index_dict[appliance].append(idx+1)
                except ValueError:
                    app_index_dict[appliance].append(-1)
                
            #if no devices found in house, remove the house and move to the next
            if np.sum(list(app_index_dict.values())) == -len(self.appliance_names):
                self.house_indicies.remove(house_id)
                continue
                
            #Read appliance data and merge  
            for appliance in self.appliance_names:
                channel_idx = app_index_dict[appliance][0]
                if channel_idx == -1:
                    house_data.insert(len(house_data.columns), appliance, np.zeros(len(house_data)))
                else:
                    channel_path      = house_folder.joinpath('channel_' + str(channel_idx) + '.dat')
                    appl_data         = pd.read_csv(channel_path, sep = ' ', header = None)
                    appl_data.columns = ['time',appliance]
                    appl_data['time'] = pd.to_datetime(appl_data['time'],unit = 's')          
                    appl_data         = appl_data.set_index('time').resample(self.sampling).mean().fillna(method = 'ffill', limit = 30)   
                    house_data        = pd.merge(house_data, appl_data, how='inner', on='time')
                                            
            if house_id == self.house_indicies[0]:
                entire_data = house_data
                if len(self.house_indicies) == 1:
                    entire_data = entire_data.reset_index(drop=True)
            else:
                entire_data = entire_data.append(house_data, ignore_index=True)
                
        entire_data                  = entire_data.dropna().copy()
        entire_data                  = entire_data[entire_data['aggregate'] > 0] #remove negative values (possible mistakes)
        entire_data[entire_data < 5] = 0 #remove very low values
        entire_data                  = entire_data.clip([0] * len(entire_data.columns), self.cutoff, axis=1) # force values to be between 0 and cutoff
        
        return entire_data.values[:, 0], entire_data.values[:, 1:]

    def compute_status(self, data):
        status = np.zeros(data.shape)
        if len(data.squeeze().shape) == 1:
            columns = 1
        else:
            columns = data.squeeze().shape[-1]

        for i in range(columns):
            initial_status = data[:, i] >= self.threshold[i]
            status_diff    = np.diff(initial_status)
            events_idx     = status_diff.nonzero()

            events_idx  = np.array(events_idx).squeeze()
            events_idx += 1

            if initial_status[0]:
                events_idx = np.insert(events_idx, 0, 0)

            if initial_status[-1]:
                events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

            events_idx     = events_idx.reshape((-1, 2))
            on_events      = events_idx[:, 0].copy()
            off_events     = events_idx[:, 1].copy()
            assert len(on_events) == len(off_events)

            if len(on_events) > 0:
                off_duration = on_events[1:] - off_events[:-1]
                off_duration = np.insert(off_duration, 0, 1000)
                on_events    = on_events[off_duration > self.min_off[i]]
                off_events   = off_events[np.roll(off_duration, -1) > self.min_off[i]]

                on_duration  = off_events - on_events
                on_events    = on_events[on_duration  >= self.min_on[i]]
                off_events   = off_events[on_duration >= self.min_on[i]]
                assert len(on_events) == len(off_events)

            temp_status = data[:, i].copy()
            temp_status[:] = 0
            for on, off in zip(on_events, off_events):
                temp_status[on: off] = 1
            status[:, i] = temp_status

        return status    
    
    def get_datasets(self):
        val_end = int(self.val_size * len(self.x))
        
        train = NILMDataset(self.x[val_end:],
                            self.y[val_end:],
                            self.status[val_end:],
                            self.window_size,
                            self.window_stride)
        
        val   = NILMDataset(self.x[:val_end],
                            self.y[:val_end],
                            self.status[:val_end],
                            self.window_size,
                            self.window_size) #non-overlapping windows

        return train, val

# RED Parser
# ----------
class Redd_Parser:
    
    def __init__(self,args):
        self.data_location   = args.data_location
        self.house_indicies  = args.house_indicies
        self.appliance_names = args.appliance_names
        self.sampling        = args.sampling
        self.normalize       = args.normalize

        
        self.cutoff          =  [args.cutoff[appl]    for appl in ['aggregate']+args.appliance_names]
        self.threshold       =  [args.threshold[appl] for appl in args.appliance_names]
        self.min_on          =  [args.min_on[appl]    for appl in args.appliance_names]
        self.min_off         =  [args.min_off[appl]   for appl in args.appliance_names]

        self.val_size        =  args.validation_size
        self.window_size     =  args.window_size
        self.window_stride   =  args.window_stride
        
        self.x, self.y       = self.load_data()
        if self.normalize == 'mean':
          self.x_mean = np.mean(self.x)
          self.x_std  = np.std(self.x)
          self.x = (self.x - self.x_mean) / self.x_std
        elif self.normalize == 'minmax':
          self.x_min = min(self.x)
          self.x_max = max(self.x)
          self.x = (self.x - self.x_min)/(self.x_max-self.x_min)
        self.status          = self.compute_status(self.y)

    
    def load_data(self):
        for appliance in self.appliance_names:
            assert appliance in ['dishwasher','refrigerator', 'microwave', 'washer_dryer']

        for house_id in self.house_indicies:
            assert house_id in [1, 2, 3, 4, 5, 6]

            directory = Path(self.data_location).joinpath('redd_lf')  

            for house_id in self.house_indicies:
                house_folder = directory.joinpath('house_' + str(house_id))
                house_label  = pd.read_csv(house_folder.joinpath('labels.dat'),    sep=' ', header=None)
                main_1       = pd.read_csv(house_folder.joinpath('channel_1.dat'), sep=' ', header=None)
                main_2       = pd.read_csv(house_folder.joinpath('channel_2.dat'), sep=' ', header=None)
                
                house_data            = pd.merge(main_1, main_2, how='inner', on=0)
                house_data.iloc[:, 1] = house_data.iloc[:,1] + house_data.iloc[:,2]
                house_data            = house_data.iloc[:, 0: 2]

                appliance_list = house_label.iloc[:, 1].values
                app_index_dict = defaultdict(list)

                for appliance in self.appliance_names:
                    data_found = False
                    for i in range(len(appliance_list)):
                        if appliance_list[i] == appliance:
                            app_index_dict[appliance].append(i + 1)
                            data_found = True

                    if not data_found:
                        app_index_dict[appliance].append(-1)

                if np.sum(list(app_index_dict.values())) == -len(self.appliance_names):
                    self.house_indicies.remove(house_id)
                    continue

                for appliance in self.appliance_names:
                    if app_index_dict[appliance][0] == -1:
                        temp_values          = house_data.copy().iloc[:, 1]
                        temp_values[:]       = 0
                        temp_data            = house_data.copy().iloc[:, :2]
                        temp_data.iloc[:, 1] = temp_values
                    else:
                        temp_data = pd.read_csv(house_folder.joinpath('channel_' + str(app_index_dict[appliance][0]) + '.dat'), sep=' ', header=None)

                    if len(app_index_dict[appliance]) > 1:
                        for idx in app_index_dict[appliance][1:]:
                            temp_data_           = pd.read_csv(house_folder.joinpath('channel_' + str(idx) + '.dat'), sep=' ', header=None)
                            temp_data            = pd.merge(temp_data, temp_data_, how='inner', on=0)
                            temp_data.iloc[:, 1] = temp_data.iloc[:,1] + temp_data.iloc[:, 2]
                            temp_data            = temp_data.iloc[:, 0: 2]

                    house_data = pd.merge(house_data, temp_data, how='inner', on=0)

                house_data.iloc[:, 0] = pd.to_datetime(house_data.iloc[:, 0], unit='s')
                house_data.columns    = ['time', 'aggregate'] + [i for i in self.appliance_names]
                house_data            = house_data.set_index('time')
                house_data            = house_data.resample(self.sampling).mean().fillna(method='ffill', limit=30)

                if house_id == self.house_indicies[0]:
                    entire_data = house_data
                else:
                    entire_data = entire_data.append(house_data, ignore_index=True)

                entire_data                  = entire_data.dropna().copy()
                entire_data                  = entire_data[entire_data['aggregate'] > 0]
                entire_data[entire_data < 5] = 0
                entire_data                  = entire_data.clip([0] * len(entire_data.columns), self.cutoff, axis=1)

            return entire_data.values[:, 0], entire_data.values[:, 1:]
    
    
    def compute_status(self, data):
        status = np.zeros(data.shape)
        if len(data.squeeze().shape) == 1:
            columns = 1
        else:
            columns = data.squeeze().shape[-1]

        for i in range(columns):
            initial_status = data[:, i] >= self.threshold[i]
            status_diff    = np.diff(initial_status)
            events_idx     = status_diff.nonzero()

            events_idx  = np.array(events_idx).squeeze()
            events_idx += 1

            if initial_status[0]:
                events_idx = np.insert(events_idx, 0, 0)

            if initial_status[-1]:
                events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

            events_idx     = events_idx.reshape((-1, 2))
            on_events      = events_idx[:, 0].copy()
            off_events     = events_idx[:, 1].copy()
            assert len(on_events) == len(off_events)

            if len(on_events) > 0:
                off_duration = on_events[1:] - off_events[:-1]
                off_duration = np.insert(off_duration, 0, 1000)
                on_events    = on_events[off_duration > self.min_off[i]]
                off_events   = off_events[np.roll(off_duration, -1) > self.min_off[i]]

                on_duration  = off_events - on_events
                on_events    = on_events[on_duration  >= self.min_on[i]]
                off_events   = off_events[on_duration >= self.min_on[i]]
                assert len(on_events) == len(off_events)

            temp_status = data[:, i].copy()
            temp_status[:] = 0
            for on, off in zip(on_events, off_events):
                temp_status[on: off] = 1
            status[:, i] = temp_status

        return status    
    
    def get_datasets(self):
        val_end = int(self.val_size * len(self.x))
        
        train = NILMDataset(self.x[val_end:],
                            self.y[val_end:],
                            self.status[val_end:],
                            self.window_size,
                            self.window_stride)
        
        val   = NILMDataset(self.x[:val_end],
                            self.y[:val_end],
                            self.status[:val_end],
                            self.window_size,
                            self.window_size) #non-overlapping windows

        return train, val

    
    

# NILM Dataloader
# --------------

class NILMDataloader():
    def __init__(self, args, data_parser):
        self.args = args
        self.batch_size = args.batch_size

        self.train_dataset, self.val_dataset = data_parser.get_datasets()

    def get_dataloaders(self):
        train_loader = self._get_loader(self.train_dataset)
        val_loader   = self._get_loader(self.val_dataset)
        return train_loader, val_loader

    def _get_loader(self, dataset):
        dataloader = data_utils.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           pin_memory=True)
        return dataloader


