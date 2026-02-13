from __future__ import division, print_function
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score
import scipy.io as scio
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

nmi = normalized_mutual_info_score
vmeasure = v_measure_score
ari = adjusted_rand_score
import h5py

def safe_load_mat(file_path):
    try:
        return scio.loadmat(file_path)
    except OSError:
        print(f"Detected MATLAB v7.3 format for {file_path}, switching to h5py...")
        # Handle v7.3 format
        mat_data = {}
        with h5py.File(file_path, 'r') as f:
            for key, value in f.items():
                # Skip MATLAB metadata references
                if isinstance(value, h5py.Group):
                    continue
                
                # Read data and transpose
                # Note: MATLAB (column-major) -> H5PY (row-major), usually requires transpose
                data = np.array(value)
                
                # Transpose only when dimension is greater than 1 to keep consistency with scipy
                if data.ndim > 1:
                    data = data.transpose()
                    
                mat_data[key] = data
        return mat_data
    except Exception as e:
        # If it is file corruption (OSError) or other issues, raise directly
        print(f"Error loading {file_path}: {e}")
        raise e

class multiViewDataset2(Dataset):

    def __init__(self, dataName,viewNumber,method,pretrain):
        dataPath = './data/' + dataName + '.mat'
        matData = safe_load_mat(dataPath)
        self.data=[]
        self.viewNumber = viewNumber
        for viewIndex in range(viewNumber):
            temp=matData['X'+str(viewIndex+1)].astype(np.float32)
            if self.viewNumber>=2:
                temp=min_max_scaler.fit_transform(temp)
            self.data.append(temp)
        Y = matData['Y'][0]
        self.labels = Y
        self.pretrain=pretrain


    def __getitem__(self, index):
        data_tensor=[]
        for viewIndex in range(self.viewNumber):
            m=self.data[viewIndex][index]
            data_tensor.append(torch.from_numpy(self.data[viewIndex][index]))
        label= self.labels[index]
        label_tensor=torch.tensor(label)
        index_tensor = torch.tensor(index)

        return data_tensor, label_tensor, index_tensor


    def __len__(self):
        return len(self.labels)

class imagedataset(Dataset):

    def __init__(self, dataName,viewNumber,method,pretrain):
        dataPath = './data/' + dataName + '.mat'
        matData = safe_load_mat(dataPath)
        self.data=[]
        self.viewNumber = viewNumber
        for viewIndex in range(viewNumber):
            temp=matData['X'+str(viewIndex+1)].astype(np.float32)
            if self.viewNumber>=6:
                temp=min_max_scaler.fit_transform(temp)
            self.data.append(temp)
        Y = matData['Y'][0]
        self.labels = Y
        self.pretrain=pretrain


    def __getitem__(self, index):
        data_tensor=[]
        for viewIndex in range(self.viewNumber):
            m=self.data[viewIndex][index]
            data_tensor.append(torch.from_numpy(self.data[viewIndex][index]))
        label= self.labels[index]
        label_tensor=torch.tensor(label)
        index_tensor = torch.tensor(index)

        return data_tensor, label_tensor, index_tensor


    def __len__(self):
        return len(self.labels)



import numpy as np
import scipy.io as scio
import random
import math
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

# Define a function to generate a mask, and set the random seed
def get_mask(num_views, data_size, missing_rate, seed=42):
    assert num_views >= 2
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    miss_sample_num = math.floor(data_size * missing_rate)
    data_ind = list(range(data_size))
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones([data_size, num_views])
    
    for j in range(miss_sample_num):
        while True:
            rand_v = np.random.rand(num_views)
            v_threshold = np.random.rand(1)
            observed_ind = (rand_v >= v_threshold)
            ind_ = ~observed_ind
            rand_v[observed_ind] = 1
            rand_v[ind_] = 0
            if 0 < np.sum(rand_v) < num_views:
                break
        mask[miss_ind[j]] = rand_v
    return mask

class imagedataset_with_mask(Dataset):
    def __init__(self, dataName, viewNumber, method, pretrain, missing_rate=0.0):

        
        # Generate mask matrix
        if missing_rate > 0:
            dataPath = './data/' + dataName + '.mat'
            matData = safe_load_mat(dataPath)
            self.data = []
            self.viewNumber = viewNumber
            self.pretrain = pretrain
            self.labels = matData['Y'][0]
            self.scaler = MinMaxScaler()
            self.data_size = matData['X1'].astype(np.float32).shape[0]

            self.mask = get_mask(viewNumber, self.data_size, missing_rate)
            # Read multi-view data and apply mask and normalization here
            for viewIndex in range(viewNumber):
                # Get data for this view
                temp = matData['X' + str(viewIndex + 1)].astype(np.float32)

                # If the number of views >= 6, perform min-max normalization
                if self.viewNumber >= 6:
                    temp = self.scaler.fit_transform(temp)
                # 1. Convert mask to a 1D boolean array
                mask = (self.mask[:, viewIndex]).squeeze()  # Remove the redundant dimension in (70000, 1) to make it (70000,)
                mask_bool = mask == 1  # Convert mask to boolean array

                # 2. Pre-create an all-zero array to store the results
                view_data_with_mask = np.zeros_like(temp)  # Create an all-zero array with the same shape as temp

                # 3. Use boolean indexing to keep the samples that meet the conditions
                view_data_with_mask[mask_bool] = temp[mask_bool]  # Only keep samples where mask is True
                self.data.append(view_data_with_mask)                
                print("viewIndex:{viewIndex}")

        else:
            # If there is no missing rate, the mask matrix is all 1s, representing that all view data exists
            dataPath = './data/' + dataName + '.mat'
            matData = safe_load_mat(dataPath)
            self.data=[]
            self.viewNumber = viewNumber
            for viewIndex in range(viewNumber):
                temp=matData['X'+str(viewIndex+1)].astype(np.float32)
                if self.viewNumber>=6:
                    temp=min_max_scaler.fit_transform(temp)
                self.data.append(temp)
            Y = matData['Y'][0]
            self.labels = Y
            self.pretrain=pretrain
            self.data_size = self.data[0].shape[0]

            self.mask = np.ones((self.data_size, viewNumber))



    def __getitem__(self, index):
        data_tensor=[]
        for viewIndex in range(self.viewNumber):
            m=self.data[viewIndex][index]
            data_tensor.append(torch.from_numpy(self.data[viewIndex][index]))
        label= self.labels[index]
        label_tensor=torch.tensor(label)
        index_tensor = torch.tensor(index)

        return data_tensor, label_tensor, index_tensor


    def __len__(self):
        return len(self.labels)


import numpy as np
from scipy.stats import entropy

# Calculate the entropy of each view, used to determine whether the view is close to all 0s
def calculate_entropy(view_data):
    """
    Calculate the entropy value of the data. Lower entropy means less diversity in the data.
    
    Parameters:
    - view_data: sample data (single view)
    
    Returns:
    - Entropy value
    """
    # If the data is all 0s, return 0 entropy directly
    if np.all(view_data == 0):
        return 0.0
    # Normalize the data into a probability distribution to avoid negative values
    normalized_view = np.histogram(view_data, bins=10, range=(view_data.min(), view_data.max()), density=True)[0]
    return entropy(normalized_view + 1e-9)  # Avoid 0 probability

from tqdm import tqdm

class imagedataset_with_mask_and_full(Dataset):
    def __init__(self, dataName, viewNumber, method, pretrain, missing_rate=0.0):
        # Generate mask matrix
        if missing_rate > 0:
            dataPath = './data/' + dataName + '.mat'
            matData = safe_load_mat(dataPath)
            self.data = []
            self.viewNumber = viewNumber
            self.pretrain = pretrain
            self.labels = matData['Y'][0]
            self.scaler = MinMaxScaler()
            self.data_size = matData['X1'].astype(np.float32).shape[0]
            self.mask = get_mask(viewNumber, self.data_size, missing_rate)
            # Read multi-view data and apply mask and normalization here
            for viewIndex in range(viewNumber):
                # Get data for this view
                temp = matData['X' + str(viewIndex + 1)].astype(np.float32)
                # # If the number of views >= 6, perform min-max normalization
                # if self.viewNumber >= 6:
                #     temp = self.scaler.fit_transform(temp)

                mask = (self.mask[:, viewIndex]).squeeze()  # Remove the redundant dimension in (70000, 1) to make it (70000,)
                mask_bool = mask == 1  # Convert mask to boolean array
                view_data_with_mask = np.zeros_like(temp)  # Create an all-zero array with the same shape as temp
                view_data_with_mask[mask_bool] = temp[mask_bool]  # Only keep samples where mask is True
                self.data.append(view_data_with_mask)                
                print(f"viewIndex:{viewIndex}")
            # Traverse all samples to handle cases where the view is all 0s or entropy is close to 0
            print("Data Fulling...")
            for i in tqdm(range(self.data_size), desc="Processing samples"):
                # pass
                # Check the view data of each sample
                for viewIndex in range(self.viewNumber):
                    current_view = self.data[viewIndex][i]  # Current view data
                    if np.all(current_view == 0) or calculate_entropy(current_view) < 1e-5:
                        # Find other unmasked views in this sample
                        available_views = [v for v in range(self.viewNumber) if not np.all(self.data[v][i] == 0)]
                        # If there are more than 1 available views, randomly select another view to replace the current one
                        if len(available_views) > 0:
                            chosen_view = np.random.choice(available_views)
                            self.data[viewIndex][i] = self.data[chosen_view][i].copy()  # Replace with a randomly selected non-zero view

        else:
            dataPath = './data/' + dataName + '.mat'
            matData = safe_load_mat(dataPath)
            self.data=[]
            self.viewNumber = viewNumber
            for viewIndex in range(viewNumber):
                temp=matData['X'+str(viewIndex+1)].astype(np.float32)
                if self.viewNumber>=6:
                    temp=min_max_scaler.fit_transform(temp)
                self.data.append(temp)
            Y = matData['Y'][0]
            self.labels = Y
            self.pretrain=pretrain
            self.data_size = self.data[0].shape[0]

            self.mask = np.ones((self.data_size, viewNumber))



    def __getitem__(self, index):
        data_tensor=[]
        for viewIndex in range(self.viewNumber):
            m=self.data[viewIndex][index]
            data_tensor.append(torch.from_numpy(self.data[viewIndex][index]))
        label= self.labels[index]
        label_tensor=torch.tensor(label)
        index_tensor = torch.tensor(index)

        return data_tensor, label_tensor, index_tensor


    def __len__(self):
        return len(self.labels)


class WKLDiv(torch.nn.Module):
    def __init__(self):
        super(WKLDiv, self).__init__()

    def forward(self, q_logit, p, w):
        p_logit=torch.log(p + 1e-12)
        kl = torch.sum(p * (p_logit- q_logit)*w, 1)
        return torch.mean(kl)


#######################################################
# Evaluate Critiron
#######################################################

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import re

def initialize_logging(args, log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    log_file_name = (f"{args.dataset}finetuning_mr{args.missing_rate}_g1{args.gamma_1}_g2{args.gamma_2}_"
                     f"g3{args.gamma_3}_g4{args.gamma_4}.log")
    
    log_file = os.path.join(log_dir, log_file_name)

    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(f"Initializing new log file: {log_file}")
    
    return log_file

def plot_pretrain_metrics(log_file, args):
    epochs, mse_losses, kl_losses, total_losses = parse_pretrain_log(log_file)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mse_losses, label='MSE Loss', color='blue')
    plt.plot(epochs, kl_losses, label='KL Loss', color='red')
    plt.plot(epochs, total_losses, label='Total Loss', color='green')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pretrain Losses')
    plt.legend()
    plt.grid(True)

    experiment_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = (
        f"{args.dataset}pretrain_metrics_mr{args.missing_rate}_g1{args.gamma_1}_g2{args.gamma_2}_"
        f"g3{args.gamma_3}_g4{args.gamma_4}_{experiment_time}.svg"
    )

    plt.savefig(output_file, format='svg')
    plt.close()

def plot_finetuning_metrics(log_file, args):
    epochs, mse_losses, mmd_losses, kl_losses, dskl_losses, total_losses = parse_finetuning_log(log_file)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mse_losses, label='MSE Loss', color='blue')
    plt.plot(epochs, mmd_losses, label='MMD Loss', color='orange')
    plt.plot(epochs, kl_losses, label='KL Loss', color='red')
    plt.plot(epochs, dskl_losses, label='DSKL Loss', color='purple')
    plt.plot(epochs, total_losses, label='Total Loss', color='green')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FineTuning Losses')
    plt.legend()
    plt.grid(True)

    experiment_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = (
        f"{args.dataset}finetuning_metrics_mr{args.missing_rate}_g1{args.gamma_1}_g2{args.gamma_2}_"
        f"g3{args.gamma_3}_g4{args.gamma_4}_{experiment_time}.svg"
    )

    plt.savefig(output_file, format='svg')
    plt.close()

def parse_pretrain_log(log_file):
    epochs = []
    mse_losses = []
    kl_losses = []
    total_losses = []

    with open(log_file, 'r') as f:
        for line in f:
            if "Pretrain" in line:
                match = re.search(r'Epoch: (\d+)', line)
                if match:
                    epoch = int(match.group(1))
                    mse_loss = float(re.search(r'MSE Loss: ([\d.]+)', line).group(1))
                    kl_loss = float(re.search(r'KL Loss: ([\d.]+)', line).group(1))
                    total_loss = float(re.search(r'Total Loss: ([\d.]+)', line).group(1))

                    epochs.append(epoch)
                    mse_losses.append(mse_loss)
                    kl_losses.append(kl_loss)
                    total_losses.append(total_loss)

    return epochs, mse_losses, kl_losses, total_losses

def parse_finetuning_log(log_file):
    epochs = []
    mse_losses = []
    mmd_losses = []
    kl_losses = []
    dskl_losses = []
    total_losses = []

    with open(log_file, 'r') as f:
        for line in f:
            if "FineTuning" in line:
                match = re.search(r'Epoch: (\d+)', line)
                if match:
                    epoch = int(match.group(1))
                    mse_loss = float(re.search(r'MSE Loss: ([\d.]+)', line).group(1))
                    mmd_loss = float(re.search(r'MMD Loss: ([\d.]+)', line).group(1))
                    kl_loss = float(re.search(r'KL Loss: ([\d.]+)', line).group(1))
                    dskl_loss = float(re.search(r'DSKL Loss: ([\d.]+)', line).group(1))
                    total_loss = float(re.search(r'Total Loss: ([\d.]+)', line).group(1))

                    epochs.append(epoch)
                    mse_losses.append(mse_loss)
                    mmd_losses.append(mmd_loss)
                    kl_losses.append(kl_loss)
                    dskl_losses.append(dskl_loss)
                    total_losses.append(total_loss)

    return epochs, mse_losses, mmd_losses, kl_losses, dskl_losses, total_losses