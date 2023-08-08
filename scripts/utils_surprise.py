# script for storing small functions related to surprise package

from tqdm.notebook import tqdm
import numpy as np


def produce_top_k(model, users, games, validation_dict, k=10, sample_size=200):
    """
    Description:
    -----------
    Function for returing list of top-k recommendation for every user
    using sampling strategy (we order only a subset of randolny choosen
    items instead of ordering all the items).
    
    Params:
    ------
    model - fitted surprise model
    users - list of users (their ids)
    games - list of games (their ids)
    validation_dict - dictionary with uid:gid as key-pair values, it
    contains one user-item pair from original validation set (with
    one item per user)
    k - the length of recommendation list
    sample_size - how many items we want to sample, instead of computing
    scores for all the items
    """
    preds_top_k = {}
    selected = games   

    for user_id in tqdm(users):
        #user_id,game_u_val,rating = row
        #games_u = explicit.loc[explicit.user_name==user_id].game_id.append(pd.Series(game_u_val))
        #diff = np.setdiff1d(games, games_u, assume_unique=False)
        if sample_size is not None:
            selected = np.random.choice(games, sample_size, replace=False)
            selected  = np.append(selected, validation_dict[user_id])

        user_preds = []
        for game_id in selected:
            r_hat = model.predict(user_id, game_id).est
            user_preds.append((r_hat,game_id))

        # selecting only top 10 ratings
        user_preds.sort(key=lambda x: x[0], reverse=True)
        # selecting only game ids
        user_preds = [i[1] for i in user_preds[:k]]

        # adding results to the dict
        preds_top_k[user_id] = user_preds
    
    return preds_top_k