import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

import scipy.sparse
import time
import pickle
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm.notebook import tqdm

from metrics import Evaluator
from utils_VAE import BaseMultiVAE, TrainableMultVAE, loss_function, naive_sparse2tensor, sparse2torch_sparse



mypath = "/home/mmarzec12/data/"
savepath = "/home/mmarzec12/models/vae/model_tuning/"

explicit = pd.read_csv(mypath+"explicit_train.csv")
validation = pd.read_csv(mypath+"leave_one_out_validation.csv")


# list with (user,item) tuples from validation set
validation_list = [(u,i) for u,i in zip(validation.user_name, validation.game_id)]
# dict with user:game key-value pairs from validation set
validation_dict = {u:i for u,i in zip(validation.user_name, validation.game_id)}

# unique games and users
unique_users = explicit.user_name.unique()
unique_games = explicit.game_id.unique()

n_users, n_items = len(unique_users), len(unique_games)

# dictonaries to map users to unique ids and vice vers
us_to_ids = {u:i for i,u in enumerate(unique_users)}
ids_to_us = {i:u for i,u in enumerate(unique_users)}

# dictonaries to map games to unique ids and vice vers
gs_to_ids = {g:i for i,g in enumerate(unique_games)}
ids_to_gs = {i:g for i,g in enumerate(unique_games)}


implicit = pd.read_csv(mypath+"implicit_train.csv")

# filtering explicit ratings: filter ratings <6 and >=1
print(f"There is {np.sum(explicit.score <= 6)} rows with score <= 6.")
explicit = explicit[explicit.score > 6]

# we join implictit and explicit rating data
joined = pd.concat([explicit, implicit])
joined = joined[["user_name", "game_id", "score"]]
# converting all interaction data to "1" 
joined["score"] = 1

# creating sparse matrix with data
row = [us_to_ids[us] for us in joined.user_name]
col = [gs_to_ids[g] for g in joined.game_id]
data = joined.score

train_data = scipy.sparse.coo_matrix((data, (row, col)), shape=(len(unique_users), len(unique_games))).tocsr()


## Tuning the beta parameter
latent_dim = 200
dim_layers = [600]
encoder_dims = [n_items] + dim_layers + [latent_dim]
decoder_dims = [latent_dim] + dim_layers[::-1] + [n_items]
n_epochs = 100
k = 10
dropout = 0.5

base_params = {"latent_dim":latent_dim,
              "encoder_dims":encoder_dims,
               "decoder_dims":decoder_dims,
               "dropout":dropout
              }

# weigth decay==0 means not used
optimizer_kwargs = {"weight_decay":0, "lr":5e-4}

train_params = {"n_epochs":n_epochs,
               "k":k,
               "optimizer_kwargs":optimizer_kwargs,
               "beta":None}

criterion = loss_function

# preparing validation data
val_data = [(us_to_ids[u], gs_to_ids[i]) for u,i in validation_list]

# beta parameters
betas = np.arange(0, 1.0001, 0.1)

counter = 1
results = []
for beta in betas:
    train_params["beta"] = beta
    model = TrainableMultVAE(encoder_dims, decoder_dims, dropout)
    optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)
    model.fit(train_data, optimizer, criterion, val_data=val_data, n_epochs=n_epochs, k=k, beta=beta)
    
    res = {}
    
    res["beta"] = beta
    res["NDCG10"] = model.NDCGs
    res["ERR10"] = model.ERRs
    res["HR10"] = model.HRs
    res["train_params"] = train_params
    res["base_params"] = base_params
    results.append(np.mean(model.NDCGs[:-10]))
    # save the obtained results
    with open(savepath+"beta/"+f"vae_{counter}", "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    counter += 1


# pick the best beta
best_beta_idx = np.argmax(np.array(results))
best_beta = betas[best_beta_idx]
train_params["beta"] = best_beta


## Tuning the number of neurons
latent_dims = [100, 200, 300, 400]
dim_layers_s = [[600], [400], [800], [1000]]

comb = []
results = []
counter = 1
for latent_dim in latent_dims:
    for dim_layers in dim_layers_s:
        comb.append((latent_dim, dim_layers))
        
        encoder_dims = [n_items] + dim_layers + [latent_dim]
        decoder_dims = [latent_dim] + dim_layers[::-1] + [n_items]
        
        base_params["encoder_dims"] = encoder_dims
        base_params["decoder_dims"] = decoder_dims
        base_params["latent_dim"] = latent_dim
        
        model = TrainableMultVAE(encoder_dims, decoder_dims, dropout)
        optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)
        model.fit(train_data, optimizer, criterion, val_data=val_data, n_epochs=n_epochs, k=k, beta=best_beta)
        
        res = {}
    
        res["encoder_dims"] = encoder_dims
        res["decoder_dims"] = decoder_dims
        res["latent_dim"] = latent_dim
        res["NDCG10"] = model.NDCGs
        res["ERR10"] = model.ERRs
        res["HR10"] = model.HRs
        res["train_params"] = train_params
        res["base_params"] = base_params
        results.append(np.mean(model.NDCGs[:-10]))
        # save the obtained results
        with open(savepath+"neurons/"+f"vae_{counter}", "wb") as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        counter += 1     


# pick the best number of neurons
best_neurons_idx = np.argmax(np.array(results))
best_latent_dim, best_dim_layers = comb[best_neurons_idx]

encoder_dims = [n_items] + best_dim_layers + [best_latent_dim]
decoder_dims = [best_latent_dim] + best_dim_layers[::-1] + [n_items]

base_params["encoder_dims"] = encoder_dims
base_params["decoder_dims"] = decoder_dims
base_params["latent_dim"] = best_latent_dim


## Tuning dropout value 
dropouts = np.arange(0, 0.80001, 0.1)

counter = 1
results = []
for dropout in dropouts:
    
    base_params["dropout"] = dropout
    
    model = TrainableMultVAE(encoder_dims, decoder_dims, dropout)
    optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)
    model.fit(train_data, optimizer, criterion, val_data=val_data, n_epochs=n_epochs, k=k, beta=best_beta)
    
    res = {}
    
    res["dropout"] = dropout
    res["NDCG10"] = model.NDCGs
    res["ERR10"] = model.ERRs
    res["HR10"] = model.HRs
    res["train_params"] = train_params
    res["base_params"] = base_params
    results.append(np.mean(model.NDCGs[:-10]))
    # save the obtained results
    with open(savepath+"dropout/"+f"vae_{counter}", "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    counter += 1



# pick best dropout value
best_dropout_idx = np.argmax(np.array(results))
best_dropout = dropouts[best_dropout_idx]
base_params["dropout"] = best_dropout





## Save best parameters

res = {}
res["train_params"] = train_params
res["base_params"] = base_params
with open(savepath+"vae_best_params", "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)




