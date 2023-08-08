import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

import scipy.sparse
import time
import pickle
import glob
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm.notebook import tqdm
from scipy.sparse import SparseEfficiencyWarning

from surprise import Dataset, Reader, accuracy
from surprise import KNNBaseline, BaselineOnly, SVD

from metrics import Evaluator
from utils_VAE import BaseMultiVAE, TrainableMultVAE, loss_function, naive_sparse2tensor, sparse2torch_sparse

from utils_surprise import produce_top_k
from my_models import ParallSynSLIM, EASE

warnings.simplefilter('ignore',SparseEfficiencyWarning)

mypath = "/home/mmarzec12/data/"
savepath = "/home/mmarzec12/models/comparison/"



explicit = pd.read_csv(mypath+"explicit_train.csv")
validation = pd.read_csv(mypath+"leave_one_out_validation.csv")
test = pd.read_csv(mypath+"leave_one_out_test.csv")
tag_matrix = scipy.sparse.load_npz(mypath+"tag_matrix.npz")

# list with (user,item) tuples from validation set
test_list = [(u,i) for u,i in zip(test.user_name, test.game_id)]
# dict with user:game key-value pairs from validation set
test_dict = {u:i for u,i in zip(test.user_name, test.game_id)}

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
implicit["score"] = 1

# filtering explicit ratings: filter ratings <6 and >=1
print(f"There is {np.sum(explicit.score <= 6)} rows with score <= 6.")
explicit_filtered = explicit[explicit.score > 6]

# we join implictit and explicit rating data
joined = pd.concat([explicit_filtered, implicit])
joined = joined[["user_name", "game_id", "score"]]
# converting all interaction data to "1" 
joined["score"] = 1

# creating sparse matrix with data
row = [us_to_ids[us] for us in joined.user_name]
col = [gs_to_ids[g] for g in joined.game_id]
data = joined.score

train_data_joined = scipy.sparse.coo_matrix((data, (row, col)), shape=(len(unique_users), len(unique_games))).tocsr()


# preparing data for MF and Item-KNN

reader = Reader(rating_scale=(1,10))
data = Dataset.load_from_df(explicit, reader)

# using whole df as trainset
trainset_explicit = data.build_full_trainset()

# some global parameters
k = 10


## MATRIX FACTORIZATION
n_factors = 5
reg = 0.005
k = 50

params = {"lr_all":0.05, "n_epochs":20, "n_factors":n_factors, "reg_all":reg}

# train the model
start_training = time.time()
mf = SVD(**params)
mf.fit(trainset_explicit)
end_training = time.time()

print("Computing top-k list for each user...")

# produce top k list for all users
start_prediction = time.time()
top_k_list = produce_top_k(model=mf, users=unique_users, games=unique_games,
                       validation_dict=test_dict,
                       k=k, sample_size=None)

end_prediction = time.time()
print("...evaluation...")
# @50
ev = Evaluator(k=k, true=test_list, predicted=top_k_list)
ev.calculate_metrics()
ndcg, err, hr = ev.ndcg, ev.err, ev.hr

res = {}
res["training_time"] =  end_training - start_training
res["prediction_time"] = end_prediction - start_prediction
res["model_name"] = "MatrixFactorization"
res["params"] = params
res["NDCG@50"] = ndcg
res["ERR@50"] = err
res["HR@50"] = hr

# @10
k = 10
ev = Evaluator(k=k, true=test_list, predicted=top_k_list)
ev.calculate_metrics()
ndcg, err, hr = ev.ndcg, ev.err, ev.hr
res["NDCG@10"] = ndcg
res["ERR@10"] = err
res["HR@10"] = hr

with open(savepath+"MatrixFactorization", "wb") as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    
## ITEM-KNN
# shrunking coefficient
lambda_ = 10
# neighborhood size
neigh_size = 5

k = 50
# min_k from knn (surprise docs for more info)
min_k_knn = 1


# setting initial options for model training
# similarity part of the model configuration
sim_options = {"name": "pearson_baseline",
              "user_based": False,
              "min_support": 1,
              "shrinkage": lambda_}

# baseline part of the model configuration
bsl_options = {"method": "sgd",
              "learning_rate": 0.01,
              "reg":0.02,
              "n_epochs":20}

# train the model
knn = KNNBaseline(k=neigh_size, min_k=min_k_knn, 
          sim_options=sim_options, 
          bsl_options=bsl_options)

start_training = time.time()
knn.fit(trainset_explicit)
knn.sim[knn.sim<0] = 0
end_training = time.time()

print("Computing top-k list for each user...")
# produce top k list for all users
start_prediction = time.time()
top_k_list = produce_top_k(model=knn, users=unique_users, games=unique_games,
                       validation_dict=test_dict,
                       k=k, sample_size=None)
end_prediction = time.time()


print("...evaluation...")
ev = Evaluator(k=k, true=test_list, predicted=top_k_list)
ev.calculate_metrics()

res = {}
res["training_time"] =  end_training - start_training
res["prediction_time"] = end_prediction - start_prediction
res["model_name"] = "ItemKNN"
res["neighborhood_size"] = neigh_size
res["min_k"] = min_k_knn
res["params_similarity"] = sim_options
res["params_baseline"] = bsl_options
res["NDCG@50"] = ev.ndcg
res["ERR@50"] = ev.err
res["HR@50"] = ev.hr

# @10
k = 10
ev = Evaluator(k=k, true=test_list, predicted=top_k_list)
ev.calculate_metrics()

res["NDCG@10"] = ev.ndcg
res["ERR@10"] = ev.err
res["HR@10"] = ev.hr

with open(savepath+"ItemKNN", "wb") as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    
## SLIM

# read params 
best_params = pd.read_pickle("/home/mmarzec12/models/slim/"+"slim_best_params")
l1_reg = best_params["l1_reg"]
l2_reg = best_params["l2_reg"]

# set the parameters
slim = ParallSynSLIM(l1_reg, l2_reg)

start_training = time.time()
# train the model
slim.fit(train_data_joined)
end_training = time.time()

# how many nonzero entires in W matirx
proc = 100*slim.W.nnz/slim.W.shape[0]**2

print("Computing top-k list for each user...")
# produce top k list for all users
start = time.time()
k = 50
start_prediction = time.time()
top_k_list = slim.calculate_top_k(train_data_joined, ids_to_gs, ids_to_us, k=k)
pred_time = time.time() - start
end_prediction = time.time()

print("...evaluation...")
ev = Evaluator(k=k, true=test_list, predicted=top_k_list)
ev.calculate_metrics()
ndcg, err, hr = ev.ndcg, ev.err, ev.hr

# save the obtained results

res = {}
res["training_time"] =  end_training - start_training
res["prediction_time"] = end_prediction - start_prediction
res["model_name"] = "SLIM"
res["l1_reg"] = l1_reg
res["l2_reg"] = l2_reg
res["NDCG@50"] = ndcg
res["ERR@50"] = err
res["HR@50"] = hr
res["W_zeros_percentage"] = proc
res["prediction_calc_time_seconds"] = pred_time

k = 10
ev = Evaluator(k=k, true=test_list, predicted=top_k_list)
ev.calculate_metrics()
ndcg, err, hr = ev.ndcg, ev.err, ev.hr
res["NDCG@10"] = ndcg
res["ERR@10"] = err
res["HR@10"] = hr

# save the obtained results
with open(savepath+"SLIM", "wb") as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    
## VAE 

best_params = pd.read_pickle("/home/mmarzec12/models/vae/model_tuning/"+"vae_best_params")
base_params = best_params["base_params"]
train_params = best_params["train_params"]

# preparing test data
test_data = [(us_to_ids[u], gs_to_ids[i]) for u,i in test_list]
   
    
# train the model
model = TrainableMultVAE(base_params["encoder_dims"], base_params["decoder_dims"], base_params["dropout"])

optimizer = optim.Adam(model.parameters(), **train_params["optimizer_kwargs"])
criterion = loss_function
start_training = time.time()
model.fit(train_data_joined, optimizer, criterion, val_data=None, n_epochs=train_params["n_epochs"],
          k=train_params["k"], beta=train_params["beta"])
end_training = time.time()

start_prediction = time.time()
ndcg, err, hr = model.predict_metrics(train_data_joined, test_data)
end_prediction = time.time()

res = {}

res["training_time"] =  end_training - start_training
res["prediction_time"] = end_prediction - start_prediction
res["model_name"] = "VAE"
res["NDCG@10"] = ndcg
res["ERR@10"] = err
res["HR@10"] = hr
res["params"] = best_params

model.k = 50
ndcg, err, hr = model.predict_metrics(train_data_joined, test_data)
res["NDCG@50"] = ndcg
res["ERR@50"] = err
res["HR@50"] = hr

with open(savepath+"VAE", "wb") as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)



## EASE
best_params = pd.read_pickle("/home/mmarzec12/models/ease/"+"ease_best_params")


reg = best_params
k = 50
gram = (train_data_joined.T @ train_data_joined)

start_training = time.time()
ease = EASE(regularization=reg)
ease.fit(gram)
end_training = time.time()

# @50
start_prediction = time.time()
recs = ease.calculate_top_k(train_data_joined, ids_to_gs, ids_to_us, k=k)
end_prediction = time.time()

ev = Evaluator(k=k, true=test_list, predicted=recs)
ev.calculate_metrics()
ndcg, err, hr = ev.ndcg, ev.err, ev.hr

res = {}

res["training_time"] =  end_training - start_training
res["prediction_time"] = end_prediction - start_prediction
res["model_name"] = "EASE"
res["NDCG@50"] = ndcg
res["ERR@50"] = err
res["HR@50"] = hr
res["regularization"] = reg

# @10
k = 10
ev = Evaluator(k=k, true=test_list, predicted=recs)
ev.calculate_metrics()
ndcg, err, hr = ev.ndcg, ev.err, ev.hr
    
res["NDCG@10"] = ndcg
res["ERR@10"] = err
res["HR@10"] = hr

with open(savepath+"EASE", "wb") as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


    
## CEASE
best_params = pd.read_pickle("/home/mmarzec12/models/cease/"+"cease_best_params")


reg = reg
alpha = best_params
k = 50
start_training = time.time()
tag_joined = scipy.sparse.vstack((train_data_joined, tag_matrix * alpha))
gram = (tag_joined.T @ tag_joined)

start_training = time.time()
cease = EASE(regularization=reg)
cease.fit(gram)
end_training = time.time()

# @50
start_prediction = time.time()
recs = cease.calculate_top_k(train_data_joined, ids_to_gs, ids_to_us, k=k)
end_prediction = time.time()

ev = Evaluator(k=k, true=test_list, predicted=recs)
ev.calculate_metrics()
ndcg, err, hr = ev.ndcg, ev.err, ev.hr

res = {}

res["training_time"] =  end_training - start_training
res["prediction_time"] = end_prediction - start_prediction
res["model_name"] = "CEASE"
res["NDCG@50"] = ndcg
res["ERR@50"] = err
res["HR@50"] = hr
res["regularization"] = reg

# @10
k = 10
ev = Evaluator(k=k, true=test_list, predicted=recs)
ev.calculate_metrics()
ndcg, err, hr = ev.ndcg, ev.err, ev.hr
    
res["NDCG@10"] = ndcg
res["ERR@10"] = err
res["HR@10"] = hr

with open(savepath+"CEASE", "wb") as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)





    
