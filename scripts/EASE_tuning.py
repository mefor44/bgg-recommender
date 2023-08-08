import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

import scipy.sparse
import time
import pickle
import glob

from tqdm.notebook import tqdm

from metrics import Evaluator
from my_models import EASE


mypath = "/home/mmarzec12/data/"
savepath = "/home/mmarzec12/models/ease/model_tuning/"

explicit = pd.read_csv(mypath+"explicit_train.csv")
validation = pd.read_csv(mypath+"leave_one_out_validation.csv")
tag_matrix = scipy.sparse.load_npz(mypath+"tag_matrix.npz")


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
implicit["score"] = 1

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
#item_matrix = user_matrix.T.copy()
#dok_matrix = user_matrix.todok()



## tuning EASE
k = 10

"""
gram = (train_data.T @ train_data)

regs = [200*i for i in range(1,16)]

counter = 1
for reg in regs:
    print(f"Model number {counter} is being trained.")
    ease = EASE(regularization=reg)
    ease.fit(gram)
    recs = ease.calculate_top_k(train_data, ids_to_gs, ids_to_us, k=k)
    
    ev = Evaluator(k=k, true=validation_list, predicted=recs)
    ev.calculate_metrics()
    ndcg, err, hr = ev.ndcg, ev.err, ev.hr
    
    res = {}
    
    res["regularization"] = reg
    res["NDCG"] = ndcg
    res["ERR"] = err
    res["HR"] = hr
    
    print(f"Saving model number {counter}...")
    with open(savepath+f"ease_{counter}", "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"... model number {counter} was saved.")
        
    counter += 1
"""    
    
    
## tuning CEASE (with side information)
savepath = "/home/mmarzec12/models/cease/model_tuning/"
k = 10


reg = pd.read_pickle("/home/mmarzec12/models/ease/"+"ease_best_params")
alphas = [1, 5, 10, 50, 100, 500, 1000, 10000]



counter = 1

for alpha in alphas:
    start = time.time()
    print(f"Model number {counter} is being trained.")
    tag_joined = scipy.sparse.vstack((train_data, tag_matrix * alpha))
    gram = tag_joined.T @ tag_joined
    
    cease = EASE(regularization=reg)
    cease.fit(gram)
    recs = cease.calculate_top_k(train_data, ids_to_gs, ids_to_us, k=k)
    
    ev = Evaluator(k=k, true=validation_list, predicted=recs)
    ev.calculate_metrics()
    ndcg, err, hr = ev.ndcg, ev.err, ev.hr
    
    res = {}
    
    res["model_name"] = "CEASE"
    res["calculation_time"] = time.time() - start
    res["alpha"] = alpha
    res["NDCG"] = ndcg
    res["ERR"] = err
    res["HR"] = hr
    
    print(f"Saving model number {counter}...")
    with open(savepath+f"cease_{counter}", "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"... model number {counter} was saved.")
        
    counter += 1

    
    
    
    
    