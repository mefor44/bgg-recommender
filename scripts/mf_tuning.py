#skrypt do fine tuningowania modellu knn item based

#!/usr/bin/env python3


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import pickle

from surprise import Dataset, Reader, SVD, BaselineOnly, accuracy
from surprise.model_selection import GridSearchCV

from time import perf_counter
from tqdm.notebook import tqdm

from utils_surprise import produce_top_k
from metrics import Evaluator

mypath = "/home/mmarzec12/data/"
savepath = "/home/mmarzec12/models/mf/model_tuning/"


explicit = pd.read_csv(mypath+"explicit_train.csv")
validation = pd.read_csv(mypath+"leave_one_out_validation.csv")


# list with (user,item) tuples from validation set
validation_list = [(u,i) for u,i in zip(validation.user_name, validation.game_id)]
# dict with user:game key-value pairs from validation set
validation_dict = {u:i for u,i in zip(validation.user_name, validation.game_id)}

# unique series for users and games
users = explicit.user_name.unique()
games = explicit.game_id.unique()


# reading data from pandas dataframe
reader = Reader(rating_scale=(1,10))
data = Dataset.load_from_df(explicit, reader)

# using whole df as trainset
trainset = data.build_full_trainset()


# Hyperparameter tuning 

# number of factors
n_factors = [5, 10, 20, 30, 50, 100]
# regularization power
regs = [0.001, 0.005, 0.01, 0.02, 0.05]
# top-10 recommendation list
k = 10 
# sample size 
sample_size = 200

params = {"lr_all":0.05, "n_epochs":20, "n_factors":20, "reg_all":0.02}

counter = 1
for n_fac in n_factors:
    for reg in regs:
        print(f"Model number {counter} is being trained.")
        # set the parameters
        params["reg_all"] = reg
        params["n_factors"] = n_fac
        
        # train the model
        mf = SVD(**params)
        
        mf.fit(trainset)
        
        print("Computing top-k list for each user...")
        # produce top k list for all users
        top_k_list = produce_top_k(model=mf, users=users, games=games,
                                   validation_dict=validation_dict,
                                   k=k, sample_size=sample_size)
        
        
        print("...evaluation...")
        ev = Evaluator(k=k, true=validation_list, predicted=top_k_list)
        ev.calculate_metrics()
        ngcg10, err10, hr10 = ev.ndcg, ev.err, ev.hr
        
        # save the obtained results
        
        res = {}
        res["n_factors"] = n_fac
        res["reg"] = reg
        res["k"] = k
        res["ndcg10"] = ngcg10
        res["err10"] = err10
        res["hr10"] = hr10
        res["sample_size"] = sample_size
        res["all_params"] = params

        
        print(ngcg10, err10, hr10)
        # save the obtained results
        with open(savepath+f"mf_{counter}", "wb") as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        counter += 1
        print("...and end.")

