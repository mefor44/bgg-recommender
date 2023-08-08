import numpy as np
import multiprocessing as mp

import scipy.sparse
import random
import itertools


from time import perf_counter
from sklearn.linear_model import SGDRegressor
from tqdm.notebook import tqdm


class EASE:
    
    def __init__(self, regularization):
        self.reg = regularization
        
        
    def fit(self, gram_matrix, verbose=False):
        start = perf_counter()
        dense_gram = gram_matrix.copy().toarray()
        diag_indices = np.diag_indices(dense_gram.shape[0])
        dense_gram[diag_indices] += self.reg 
        P = np.linalg.inv(dense_gram)
        W = P / (-np.diag(P))
        W[diag_indices] = 0
        self.W = W
        end = perf_counter()
        if verbose:
            print(f"Training the model took {round((end - start)/60,2)} minutes.")
            
            
    def fit2(self, gram_matrix, verbose=False):
        start = perf_counter()
        dense_gram = gram_matrix.copy().toarray()
        diag_indices = np.diag_indices(dense_gram.shape[0])
        dense_gram[diag_indices] += self.reg 
        P = np.linalg.inv(dense_gram)
        W = P / (-np.diag(P))
        W[diag_indices] = 0
        self.W = W
        end = perf_counter()
        if verbose:
            print(f"Training the model took {round((end - start)/60,2)} minutes.")

        
    def calculate_top_k(self, A, ids_to_gs, ids_to_us, k=10):
        """
        Description:
        -----------
        Funtion for calculating list of top-k recommendations for
        all the users.
        
        
        Params:
        -----------
        A - sparse csr user-item interaction matrix
        
        ids_to_gs - dict mapping rows of matrix A
        to the game_ids (row_id:game_id)
        
        ids_to_us - dict mapping rows of matrix A
        to the user_ids (row_id:user_id)
        
        k - the length of  recommendation list
        
        
        Returns:
        ----------
        A dictionary with original user_ids as keys
        and list of item_ids as values. 
        """
        
        topk = {}
        #scores = A @ self.W
        #print(scores.shape)
        
        for uid in tqdm(range(len(ids_to_us.keys()))):
            r = np.array(A[uid] @ self.W).flatten()
            recs = []
            known_items = set(A[uid].indices)
            #print(r.shape)
    
            for gid in r.argsort()[::-1]:
                #print(gid)
                if gid not in known_items:
                    recs.append(ids_to_gs[gid])
                if len(recs) >= k:
                    break
    
            topk[ids_to_us[uid]] = recs
        
        return topk



class ParallSynSLIM:
    
    def __init__(self, l1_reg=0.001, l2_reg=0.0001):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
        
    def fit_one(self, A, idx_start, idx_end, alpha, l1_ratio):
        
        model = SGDRegressor(penalty='elasticnet',
                     fit_intercept=False,
                     alpha=alpha,
                     l1_ratio=l1_ratio
                     ) 
        res = []
        
        for idx in range(idx_start, idx_end):
            aj = A[:,idx].copy()
            A[:,idx] = 0
        
            model.fit(A, aj.toarray().ravel())
        
            # refill the values in data matrix
            A[:, idx] = aj
            
            # forcing non-negativity
            ws = model.coef_
            ws[ws<0] = 0
            
            # return nonzero coeffs
            for row in ws.nonzero()[0]:
                res.append((row, idx, ws[row]))
        
        return res
        
    def fit(self, A):
        print(f"start = {perf_counter()}")
        # for sklearn function
        alpha = self.l1_reg + self.l2_reg
        l1_ratio = self.l1_reg / alpha
        
        # preparing for multiprocessing
        cpu_num = mp.cpu_count()
        n_games = A.shape[1]//10
        step = n_games / cpu_num
        idx1 = [int(i*step) for i in range(cpu_num)]
        idx2 = [int(i*step) for i in range(1, cpu_num+1)]
        if idx2[-1] != n_games:
            idx2[-1] = n_games
        
        # setting pools
        pool = mp.Pool(mp.cpu_count())
        global_start = perf_counter()
        
        result = pool.starmap(self.fit_one, 
                              [(A, idx_start, idx_end, alpha, l1_ratio) for idx_start,idx_end in zip(idx1, idx2)])
        # changing the format of the outpput to build sparse coefficient matrix
        print(f"end = {perf_counter()}")
        result = list(itertools.chain(*result)) 
        row, col, data = [], [], []
        for r,c,d in result:
            row.append(r)
            col.append(c)
            data.append(d)
            
        
        self.W = scipy.sparse.coo_matrix((data, (row,col)), shape=(A.shape[1], A.shape[1]))
        
        pool.close()
        pool.join()
        
        print(f"Learning all {self.W.shape[0]} vectors took {round((perf_counter() - global_start)/60, 2)} minutes.")
        proc = 100*self.W.nnz/self.W.shape[0]**2
        print(f"In W matrix we have {self.W.nnz} nonzero elements ({round(proc,3)}%).")
        
        
    def calculate_top_k(self, A, ids_to_gs, ids_to_us, k=10):
        """
        Description:
        -----------
        Funtion for calculating list of top-k recommendations for
        all the users.
        
        
        Params:
        -----------
        A - sparse csr user-item interaction matrix
        
        ids_to_gs - dict mapping rows of matrix A
        to the game_ids (row_id:game_id)
        
        ids_to_us - dict mapping rows of matrix A
        to the user_ids (row_id:user_id)
        
        k - the length of  recommendation list
        
        
        Returns:
        ----------
        A dictionary with original user_ids as keys
        and list of item_ids as values. 
        """
        
        topk = {}
        self.W = self.W.tocsr()
        #scores = A @ self.W.tocsr()
        
        
        for uid in tqdm(range(A.shape[0])):
            scores = A[uid] @ self.W.tocsr()
            
            r = scores.toarray().flatten()
            recs = []
            known_items = set(A[uid].indices)
    
            for gid in r.argsort()[::-1]:
                if gid not in known_items:
                    recs.append(ids_to_gs[gid])
                if len(recs) >= k:
                    break
    
            topk[ids_to_us[uid]] = recs
        
        return topk



class MyOwnMatrixFactorization:

    def __init__(self, lr=0.001, lambda_=0.05, n_epochs=10, n_factors=10):
        self.lr = lr
        self.lambda_ = lambda_  # regularization power
        self.n_epochs = n_epochs
        self.n_factors = n_factors

    def split(self, X, val_size=0.2, verbose=False):
        """
        Function for splitting train data for 2 disjoint sets,
        following the rule that we select val_size*(number of user ratings)
        values for each user in validation set.
        """
        import scipy

        X = X.tocsr()
        train = scipy.sparse.lil_matrix(X.shape, dtype=np.float32)
        val = scipy.sparse.lil_matrix(X.shape, dtype=np.float32)

        # if we want to display progress bar:
        if verbose:
            from tqdm import tqdm
            print("Spliting dataset...")
            iterator = tqdm(range(X.shape[0]))
        else:
            iterator = range(X.shape[0])

        # we iterate through each user
        for u in iterator:
            # select "users row"
            x = X[u]
            # get the number of values (ratings) for given user
            n = x.nnz
            # shuffle the ids
            ids = x.indices.copy()
            np.random.shuffle(ids)
            # set train and validation ids
            ids_val = ids[:round(n*val_size)]
            ids_train = ids[round(n*val_size):]

            # append values to train and val matrices
            train[u, ids_train] = x[0,ids_train]
            val[u, ids_val] = x[0,ids_val]

        return train, val


    def MSE(self, X):
        X = X.tocoo()
        """
        Takes coo sparse matrix as input and calculate MSE
        for the given data.
        """
        loss = 0
        for u,i,r in zip(X.row, X.col, X.data):
            loss += (r - (self.mu + self.bu[u] + self.bi[i] + np.dot(self.pu[u,], self.qi[i,])))**2
        return loss / X.nnz


    def update_model_params(self, pu, qi, bu, bi, mu):
        self.pu = pu
        self.qi = qi
        self.bu = bu
        self.bi = bi
        self.mu = mu
    
    def calculate_rating(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + np.dot(self.pu[u], self.qi[i])
    

    def fit(self, X, verbose=True, validation=False, validation_size=0.2):
        from scipy.sparse import coo_matrix, lil_matrix
        from time import perf_counter
        import random

        big_start = perf_counter()
        n_users = X.shape[0]
        n_items = X.shape[1]
        self.n_users = n_users
        self.n_items = n_items
        lr = self.lr
        n_factors = self.n_factors
        lambda_ = self.lambda_
        n_epochs = self.n_epochs
        # model parameters
        pu = np.random.normal(size=(n_users, n_factors))
        qi = np.random.normal(size=(n_items, n_factors))
        mu = 0
        bu = np.random.normal(size=n_users)
        bi = np.random.normal(size=n_items)

        # losses
        loss = []
        mse_train = []
        # convert X to coo_matrix and potentialy make a validation set
        if validation:
            mse_val = []
            print("Preparing validation set...")
            train, val = self.split(X, validation_size)
            # we have to  change the format to scipy.sparse.coo_matrix
            #train, val = train.tocoo(), val.tocoo()
            train, val = train.tolil(), val.tolil()
            print("...validation set created.")
        else:
            #train = X.tocoo()
            train = X.tolil()
        
        # create idx list of pairs to iterate randomly
        idx_list = [(u,i) for u,i in zip(train.nonzero()[0], train.nonzero()[1])]
        self.X = train
        for epoch in range(n_epochs):
            if verbose:
                print(f'Processsing epoch: {epoch+1}')
                start = perf_counter()

            #  stochastic gradient descent
            j = 0
            # (for deterministc coo: iterate zip(B.row, B.col, B.data))
            random.shuffle(idx_list)
            for u,i in idx_list:
                r = train[u,i]
                error_ui = r - (mu + bu[u] + bi[i] + np.dot(pu[u], qi[i]))
                
                # update pu
                pu[u,] += lr*(qi[i,]*error_ui - lambda_*pu[u])
                # update qi
                qi[i,] += lr*(pu[u,]*error_ui - lambda_*qi[i])
                
                # update user bias
                bu[u] += lr*(error_ui - lambda_*bu[u])
                
                # update movie bias
                bi[i] += lr*(error_ui - lambda_*bi[i])
                
                # update global bias
                mu += lr*(error_ui)

                j += 1
                if j%1_000_000==0:
                    print(f"We went through {j} examples in current epoch")
            
            if verbose:
                end = perf_counter()
                print(f"Updating weights took {round((end-start)/60,2)} minutes, we iterated through {j} examples.")
            
            # adding obtained parameters to function attributes
            self.update_model_params(pu=pu, qi=qi, bu=bu, bi=bi, mu=mu)
            
            # after full epoch calculate and update loss
            current_mse_train = self.MSE(train)
            mse_train.append(current_mse_train)
            if validation:
                current_mse_val = self.MSE(val)
                mse_val.append(current_mse_val)
            
            total_loss = lambda_* (np.linalg.norm(pu) + np.linalg.norm(qi) + np.linalg.norm(bu) + np.linalg.norm(bi))
            total_loss += current_mse_train * train.nnz
            loss.append(total_loss)

            if verbose:
                end = perf_counter()
                elapsed = end - start
                print(f"Training loss: {total_loss}, running this epoch took {round(elapsed/60, 2)} minutes")
                print("-"*60)
        

        # loss includes regularization term, mse_train does not
        # moreover mse is average and loss is sum
        self.loss = loss
        self.mse_train = mse_train

        if validation:
            self.mse_val = mse_val

        if verbose:
            big_end = perf_counter()
            print(f"End of training, the whole process took {round((big_end - big_start)/60, 2)} minutes.")
        return

    def predict_top_k(self, user_ids, k):
        Xlil = self.X.tolil()
        d = {user_id: None for user_id in user_ids}
        for user_id in user_ids:
            known = np.array(Xlil.rows[user_id])
            ratings = np.array([self.calculate_rating(u=user_id, i=i) for i in range(self.n_items)])
            ratings = ratings[~known]
            sorted_ids_k = np.argsort(-ratings)[:k]
            d[user_id] = sorted_ids_k
        return d