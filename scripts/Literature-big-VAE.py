import pandas as pd
import numpy as np
from time import perf_counter
from tqdm import tqdm
import scipy 
import pickle
import glob

import scipy.sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from metrics import Evaluator, NDCG, NDCG_user, NDCG_MAP, NDCG_AP_user
from my_models import EASE, ParallSynSLIM
from utils_VAE import BaseMultiVAE, TrainableMultVAE, loss_function, naive_sparse2tensor, sparse2torch_sparse


train_data = scipy.sparse.load_npz("/home/mmarzec12/data/"+"train_data_big.npz")
savepath = "/home/mmarzec12/data/"
true = pd.read_pickle(savepath+"test_data_big")
val_dict = pd.read_pickle(savepath+"val_data_big")

n_items = train_data.shape[1]


best_params = pd.read_pickle("/home/mmarzec12/models/vae/model_tuning/"+"vae_best_params")

base_params = best_params["base_params"]
base_params["encoder_dims"][0] = n_items
base_params["decoder_dims"][-1] = n_items
train_params = best_params["train_params"]
train_params["n_epochs"] = 15
#train_params["beta"]= 0.2

model = TrainableMultVAE(base_params["encoder_dims"], base_params["decoder_dims"], base_params["dropout"])

optimizer = optim.Adam(model.parameters(), **train_params["optimizer_kwargs"])
criterion = loss_function
start_time = perf_counter()
model.fit(train_data, optimizer, criterion, val_data=val_dict, n_epochs=train_params["n_epochs"],
          k=train_params["k"], beta=train_params["beta"], leave_one_out=False)
end_time = perf_counter()

recs = model.predict_dict(train_data)

k = 10
dict_scores = NDCG_MAP(k, true, recs)

res = {}
res["model_name"] = "VAE"
res["min_n_users"] = 8
res["min_n_items"] = 8
res["NDCGs_val"] = model.NDCGs
res["NDCG"] = dict_scores["NDCG"]
res["MAP"] = dict_scores["MAP"]
res["n_epochs"] = train_params["n_epochs"]
res["k"] = k
res["training_time"] = end_time - start_time


with open("/home/mmarzec12/models/comparison_literature/"+"VAE_8", "wb") as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)









