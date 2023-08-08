import pandas as pd
import numpy as np
from time import perf_counter
from tqdm import tqdm
import scipy 
import pickle
import glob
import warnings

import scipy.sparse

from scipy.sparse import SparseEfficiencyWarning

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from metrics import Evaluator, NDCG, NDCG_user, NDCG_MAP, NDCG_AP_user
from my_models import EASE, ParallSynSLIM
from utils_VAE import BaseMultiVAE, TrainableMultVAE, loss_function, naive_sparse2tensor, sparse2torch_sparse

warnings.simplefilter('ignore',SparseEfficiencyWarning)
train_data = scipy.sparse.load_npz("/home/mmarzec12/data/"+"train_data_big.npz")
savepath = "/home/mmarzec12/data/"
true = pd.read_pickle(savepath+"test_data_big")

n_items = train_data.shape[1]


best_params = pd.read_pickle("/home/mmarzec12/models/slim/"+"slim_best_params")
l1_reg = best_params["l1_reg"]
l2_reg = best_params["l2_reg"]
k = 10

slim = ParallSynSLIM(l1_reg, l2_reg)
        
# train the model
start = perf_counter()
slim.fit(train_data)
end = perf_counter()
print(f"Time elapsed = {round((end-start)/60, 2)} minutes.")
elapsed = end - start

recs = slim.calculate_top_k(train_data, ids_to_gs, ids_to_us, k=k)

tmp = {us_to_ids[us]:[gs_to_ids[g] for g in games] for us,games in recs.items()}
recs = tmp

dict_scores = NDCG_MAP(k, true, recs)

res = {}
res["model_name"] = "SLIM"
res["min_n_users"] = 8
res["min_n_items"] = 8
res["model_name"] = "SLIM"
res["NDCG"] = dict_scores["NDCG"]
res["MAP"] = dict_scores["MAP"]
res["k"] = k
res["training_time"] = elapsed


with open("/home/mmarzec12/models/comparison_literature/"+"SLIM_8", "wb") as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)









