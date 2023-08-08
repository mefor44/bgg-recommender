#skrypt do fine tuningowania modellu knn item based

#!/usr/bin/env python3


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import pickle

from surprise import Dataset, Reader, KNNBaseline, BaselineOnly, accuracy
from surprise.model_selection import GridSearchCV
from metrics import Evaluator
from utils_surprise import produce_top_k

from time import perf_counter
from tqdm.notebook import tqdm


mypath = "/home/mmarzec12/data/"
savepath = "/home/mmarzec12/models/knn/model_tuning/"


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

# shrunking coefficient
lambdas = [10, 30, 50]
# neighborhood size
negihborhood_sizes = [3, 5, 10, 20, 30]
# top-10 recommendation list
k = 10 
# sample size 
sample_size = 200
# min_k from knn (surprise docs for more info)
min_k_knn = 1


# setting initial options for model training
# similarity part of the model configuration
sim_options = {"name": "pearson_baseline",
              "user_based": False,
              "min_support": 1,
              "shrinkage": 10}
# baseline part of the model configuration
bsl_options = {"method": "sgd",
              "learning_rate": 0.01,
              "reg":0.02,
              "n_epochs":20}

counter = 1
for lambda_ in lambdas:
    for neigh_size in negihborhood_sizes:
        print(f"Model number {counter} is being trained.")
        # set the parameters
        sim_options["shrinkage"] = lambda_
        
        # train the model
        knn = KNNBaseline(k=neigh_size, min_k=min_k_knn, 
                  sim_options=sim_options, 
                  bsl_options=bsl_options)
        
        knn.fit(trainset)
        knn.sim[knn.sim<0] = 0
        
        print("Computing top-k list for each user...")
        # produce top k list for all users
        top_k_list = produce_top_k(model=knn, users=users, games=games,
                                   validation_dict=validation_dict,
                                   k=k, sample_size=sample_size)
        
        
        print("...evaluation...")
        ev = Evaluator(k=k, true=validation_list, predicted=top_k_list)
        ev.calculate_metrics()
        ngcg10, err10, hr10 = ev.ndcg, ev.err, ev.hr
        
        # save the obtained results
        res = {}
        res["shrunking_coef"] = lambda_
        res["neighborhood_size"] = neigh_size
        res["k"] = k
        res["ndcg"] = ngcg10
        res["err"] = err10
        res["hr10"] = hr10
        res["sample_size"] = sample_size
        res["sim_options"] = sim_options
        res["bsl_options"] = bsl_options
        res["min_k_knn"] = min_k_knn
        
        # save the obtained results
        with open(savepath+f"knn_{counter}", "wb") as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        counter += 1
        print("...and end.")





