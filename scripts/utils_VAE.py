# Script with utils for VAE, model class and useful functions

import pandas as pd
import numpy as np

import scipy.sparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter

from tqdm.notebook import tqdm
from metrics import Evaluator, NDCG, NDCG_user



class BaseMultiVAE(nn.Module):

    def __init__(self, encoder_layers, decoder_layers, dropout):
        super().__init__()
        
        if encoder_layers[-1] != decoder_layers[0]:
            raise ValueError('Output from Encoder must have the same dimension as input to the Decoder.')
        
        
        # last layer of encoder is for both mean and variance
        tmp_encoder_layers = encoder_layers.copy()
        tmp_encoder_layers[-1] = tmp_encoder_layers[-1]*2
        
        self.encoder_dims = tmp_encoder_layers
        self.decoder_dims = decoder_layers
        self.latent_dim = decoder_layers[0]
        
        self.encoder_layers = self.initialize_layers(tmp_encoder_layers)
        self.decoder_layers = self.initialize_layers(decoder_layers)

        
        self.dropout_ = nn.Dropout(dropout)
    
    
    def initialize_layers(self, layers, nonlinearity="tanh"):
        res = []
        
        for i in range(len(layers)-1):
            layer = nn.Linear(layers[i], layers[i+1])
            self.init_weights(layer)
            res.append(layer)
            
            if i != len(layers)-2:
                res.append(nn.Tanh())

        return nn.Sequential(*res)
    
    
    def forward(self, input_):
        mu, logvar = self.encode(input_)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def encode(self, input_):
        x = F.normalize(input_)
        x = self.dropout_(x)
        
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if i == len(self.encoder_layers) - 1:
                mu = x[:, :self.encoder_dims[-1]//2]
                logvar = x[:, self.encoder_dims[-1]//2:]
                
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # when in training mode we sample from 
        # normal distribution
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            # if not in training mode we want to
            # get the mu as reparametrization
            return mu
    
    def decode(self, z):
        x = z
        for layer in self.decoder_layers:
            x = layer(x)
        return x

    def init_weights(self, layer):
        # Xavier Initialization for weights
        size = layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0/(fan_in + fan_out))
        layer.weight.data.normal_(0.0, std)

        # Normal Initialization for Biases
        layer.bias.data.normal_(0.0, 0.001)

            


class TrainableMultVAE(BaseMultiVAE):
    
    def __init__(self, encoder_layers, decoder_layers, dropout):
        super().__init__(encoder_layers, decoder_layers, dropout)
        
        
    def fit(self, train_data, optimizer, criterion, n_epochs=100, batch_size=256, total_anneal_steps=200000,
              anneal_cap=0.2, log_interval=100, k=10, val_data=None, beta=0.2, leave_one_out=True):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.validation = True if val_data is not None else False
        self.optimizer = optimizer
        self.criterion = criterion
        self.k = k
        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = anneal_cap
        self.log_interval = log_interval
        self.n_users, self.n_items = train_data.shape
        self.NDCGs = []
        self.ERRs = []
        self.HRs = []
        self.beta = beta
        self.training_time_seconds = []
        self.validating_time_seconds = []
        
        idxlist = list(range(self.n_users))
        update_count = 0
        
        for epoch in range(1, n_epochs+1):
            np.random.shuffle(idxlist)
            
            epoch_start_time = time.time()
            print("Training phase...")
            self.train_one_epoch(train_data, idxlist, epoch_num=epoch, update_count=update_count)
            print(f"Training took {round(time.time() - epoch_start_time,2)} seconds.")
            self.training_time_seconds.append(round(time.time() - epoch_start_time,2))
            
            if self.validation:
                print("Evaluation phase...")
                val_epoch_start_time = time.time()
                if leave_one_out:
                    ndcg, err, hr = self.validate(train_data, val_data)
                else:
                    recs = self.predict_dict(train_data)
                    ndcg = NDCG(self.k, val_data, recs)
                    self.NDCGs.append(ndcg)
                    err, hr = 0.0000, 0.0000
                    
                print(f"Validating took {round(time.time() - val_epoch_start_time,2)} seconds.")
                self.validating_time_seconds.append(round(time.time() - val_epoch_start_time,2))
            
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:4.2f}s | '
                        'NDCG@10 {:5.3f} | ERR@10 {:5.3f} | HR@10 {:5.3f}'.format(
                            epoch, time.time() - epoch_start_time, ndcg, err, hr))
                print('-' * 89)
            

    def train_one_epoch(self, train_data, idxlist, epoch_num, update_count):
        
        self.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, start_idx in enumerate(range(0, self.n_users, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, self.n_users)
            
            # selecting appriopriate chunk of data and conveting 
            # it to tensors
            data = train_data[idxlist[start_idx:end_idx]]
            data = naive_sparse2tensor(data)
            
            # annealing
            """
            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 
                                1. * update_count / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap
            
            # record beta parameters
            self.betas[epoch_num].append(anneal)
            """
            anneal = self.beta

            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.forward(data)

            loss = self.criterion(recon_batch, data, mu, logvar, anneal)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            update_count += 1
            
            if batch_idx % self.log_interval == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                        'loss {:4.2f}'.format(
                            epoch_num, batch_idx, len(range(0, self.n_users, self.batch_size)),
                            elapsed * 1000 / self.log_interval,
                            train_loss / self.log_interval))
            
                start_time = time.time()
                train_loss = 0.0
            
    def predict_dict(self, train_data):
        # Evaluation phase
        self.eval()
        val_res = {}
        unique_users = list(range(self.n_users))
        with torch.no_grad():
            for start_idx in range(0, self.n_users, self.batch_size):
                end_idx = min(start_idx+self.batch_size, self.n_users)
                
                data = train_data[start_idx:end_idx]
                data_tensor = naive_sparse2tensor(data)
                
                # predict
                pred, mu, logvar = self.forward(data_tensor)
                # exclude examples from train set
                pred[data.nonzero()] = -float("Inf")
                _, rec = torch.topk(pred, self.k, dim=-1)
                
                # append the results
                uid = start_idx
                for u_rec in rec.numpy():
                    val_res[uid] = u_rec
                    uid+=1
        
        return val_res 
        
        
    def validate(self, train_data, val_data):
        """
        val_data - list with n_users tuples of (user_id, item_id) 
        """
        
        # Evaluation phase
        self.eval()
        val_res = {}
        unique_users = list(range(self.n_users))
        with torch.no_grad():
            for start_idx in range(0, self.n_users, self.batch_size):
                end_idx = min(start_idx+self.batch_size, self.n_users)
                
                data = train_data[start_idx:end_idx]
                data_tensor = naive_sparse2tensor(data)
                
                # predict
                pred, mu, logvar = self.forward(data_tensor)
                # exclude examples from train set
                pred[data.nonzero()] = -float("Inf")
                _, rec = torch.topk(pred, self.k, dim=-1)
                
                # append the results
                uid = start_idx
                for u_rec in rec.numpy():
                    val_res[uid] = u_rec
                    uid+=1
        
        # evaluation 
        ev = Evaluator(k=self.k, true=val_data, predicted=val_res)
        ev.calculate_metrics()
        ndcg, err, hr = ev.ndcg, ev.err, ev.hr
        self.NDCGs.append(ndcg)
        self.ERRs.append(err)
        self.HRs.append(hr)
        return ndcg, err, hr
    
    def predict_metrics(self, train_data, test_data):
        ndcg, err, hr = self.validate(train_data, test_data)
        return ndcg, err, hr
    
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    def save_model_params(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
 

            
def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t