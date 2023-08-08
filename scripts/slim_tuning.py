import pandas as pd
import numpy as np
import multiprocessing as mp
import scipy.sparse
import warnings
import pickle

from sklearn.linear_model import SGDRegressor
from time import perf_counter
from tqdm.notebook import tqdm
from scipy.sparse import SparseEfficiencyWarning


from my_models import ParallSynSLIM
from metrics import Evaluator


warnings.simplefilter('ignore',SparseEfficiencyWarning)

mypath = "/home/mmarzec12/data/"
savepath = "/home/mmarzec12/models/slim/model_tuning/"

explicit = pd.read_csv(mypath+"explicit_train.csv")
validation = pd.read_csv(mypath+"leave_one_out_validation.csv")


# list with (user,item) tuples from validation set
validation_list = [(u,i) for u,i in zip(validation.user_name, validation.game_id)]
# dict with user:game key-value pairs from validation set
validation_dict = {u:i for u,i in zip(validation.user_name, validation.game_id)}

# unique games and users
unique_users = explicit.user_name.unique()
unique_games = explicit.game_id.unique()

# dictonaries to map users to unique ids and vice vers
us_to_ids = {u:i for i,u in enumerate(unique_users)}
ids_to_us = {i:u for i,u in enumerate(unique_users)}

# dictonaries to map games to unique ids and vice vers
gs_to_ids = {g:i for i,g in enumerate(unique_games)}
ids_to_gs = {i:g for i,g in enumerate(unique_games)}


implicit = pd.read_csv(mypath+"implicit_train.csv")

# filtering explicit ratings: filter ratings <6 and >=1
print(f"There is {np.sum(explicit.score <= 6)} rows with score < 6.")
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

A = scipy.sparse.coo_matrix((data, (row, col)), shape=(len(unique_users), len(unique_games)))
A = A.tocsr()


# Hyperparameter tuning 

# number of factors
l1_regs = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05] # 0.00001, 0.0001, 
# regularization power
l2_regs = [0.00001, 0.0001, 0.001, 0.01]
# top-10 recommendation list
k = 10


counter = 1
for l1_reg in l1_regs:
    for l2_reg in l2_regs:
        print(f"Model number {counter} is being trained.")
        # set the parameters
        slim = ParallSynSLIM(l1_reg, l2_reg)
        
        # train the model
        slim.fit(A)
        
        # how many nonzero entires in W matirx
        proc = 100*slim.W.nnz/slim.W.shape[0]**2
        
        print("Computing top-k list for each user...")
        # produce top k list for all users
        start = perf_counter()
        top_k_list = slim.calculate_top_k(A, ids_to_gs, ids_to_us, k=k)
        pred_time = perf_counter() - start
        
        print("...evaluation...")
        ev = Evaluator(k=k, true=validation_list, predicted=top_k_list)
        ev.calculate_metrics()
        ngcg10, err10, hr10 = ev.ndcg, ev.err, ev.hr
        
        # save the obtained results
        
        res = {}
        res["l1_reg"] = l1_reg
        res["l2_reg"] = l2_reg
        res["k"] = k
        res["ndcg10"] = ngcg10
        res["err10"] = err10
        res["hr10"] = hr10
        res["W_zeros_percentage"] = proc
        res["prediction_calc_time_seconds"] = pred_time

        
        print(ngcg10, err10, hr10)
        # save the obtained results
        with open(savepath+f"slim_{counter}", "wb") as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        counter += 1
        print("...and end.")
