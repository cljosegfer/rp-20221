#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:47:26 2022

@author: jose
"""

# cortesia do vitor (y)
# https://github.com/vcaitite/pattern-recognition-exercises
# dps eu implemento ;(

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal
from gg import lara_graph as gabrielgraph

def euclidean_pair_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def euclidean_dist(X, x):
    return np.sqrt(np.sum((X-x)**2,axis=1))

class chip:
    def __init__(self, adj: np.ndarray = None) -> None:
        self.adj = adj

    def gg(self, X: np.ndarray) -> None:
        #gg_vec = GG_calculation(X)
        #self.adj = vec_to_adj(gg_vec, n=X.shape[0])
        self.adj = gabrielgraph(X)

    def build_pairs(self) -> None:
        pairs = np.transpose(np.array(np.where(self.adj==1)))
        all_neighbors = [[] for i in range(len(self.adj))]
        for i in range(len(pairs)):
            all_neighbors[pairs[i,0]].append(pairs[i,1])

        all_neighbors = [np.unique(all_neighbors[i]) for i in range(len(all_neighbors))]
        self.all_neighbors = all_neighbors

        original_samples = []
        neighbors = []
        for i in range(len(self.adj)):
            n_neighbors = len(all_neighbors[i])
            original_samples.extend(np.repeat(i,n_neighbors))
            neighbors.extend(all_neighbors[i])
        hyper_pairs = np.c_[original_samples,neighbors]
        hyper_pairs = np.sort(hyper_pairs, axis=1)
        hyper_pairs = np.unique(hyper_pairs, axis=0)
        self.hyper_pairs = hyper_pairs
    
    def find_boundhyper(self, y: np.ndarray) -> None:
        bound_hyper = []
        for i in range(len(self.hyper_pairs)):
            if y[self.hyper_pairs[i][0]]!=y[self.hyper_pairs[i][1]]:
                bound_hyper.append(self.hyper_pairs[i].tolist())
        self.bound_hyper = bound_hyper

    '''
    This function receives the graph and its labels and filter the noise using an
    vertex quality rule. It returns an vector containing the indexes of the noisy data points.
  
    The quality of the vertex is the number of connections(degree) between same label vertexes
    and number of all connections of the vertex.
  
    The threshold that say if the vertex is an noise is the sum of all qualities of the vertexes
    belonging to the same class, divided by the length of the current class.
    ( this will give us two thresholds )
    '''
    def filter_chipclas(self, X: np.ndarray, y: np.ndarray) -> None:
        c = []
        for i in range(len(X)):
            self.all_neighbors[i] = np.unique(self.all_neighbors[i])
            if y[i] == 1:
                c.append(np.sum(y[self.all_neighbors[i]]==1)/len(self.all_neighbors[i]))
            else:
                c.append(np.sum(y[self.all_neighbors[i]]!=1)/len(self.all_neighbors[i]))
        c = np.array(c)

        t1 = np.mean(c[y==1])
        t2 = np.mean(c[y!=1])

        valid_samples = []
        for i in range(len(X)):
            if(y[i]==1):
                if(c[i] >= t1):
                    valid_samples.append(1)
                else:
                    valid_samples.append(0)
            else:
                if(c[i] >= t2):
                    valid_samples.append(1)
                else:
                    valid_samples.append(0)
        self.valid_samples = valid_samples

    def fit(self, X: np.ndarray, y: np.ndarray, use_filter: bool = False) -> None:
        self.gg(X)
        self.build_pairs()
        if use_filter:
            self.filter_chipclas(X, y)
            valid_index = np.where(np.array(self.valid_samples)==1)[0]
            self.X_train = X[valid_index]
            self.y_train = y[valid_index]
            self.gg(self.X_train)
            self.build_pairs()
        else:
            self.X_train = X
            self.y_train = y
            self.valid_samples = np.ones(len(y))
        self.find_boundhyper(self.y_train)
        self.midpoints = []
        for i in range(len(self.bound_hyper)):
            self.midpoints.append((self.X_train[self.bound_hyper[i][0]]+self.X_train[self.bound_hyper[i][1]])/2)
        
    # Gaussian function
    def kgaussian(self, u, h):
        K = 1/(np.sqrt(2*np.pi)*h) * np.exp(-0.5*(u**2))
        return K   
        
    # My KDE Implementation
    def my_kde(self, x, X_bound, indexes_to_train, h):
        bound_elements = [item for sublist in self.bound_hyper for item in sublist]
        # print(bound_elements)
        N_train = X_bound.shape[0]
        n_train = X_bound.shape[1]
        px = np.zeros(N_train)
        K_total = 0
        for i in indexes_to_train:
            idx = np.where(self.X_train[:,:] == X_bound[i, :])
            idx = idx[0][0]
            pm_idx = int(np.where(bound_elements == idx)[0][0] / 2) # aqui peguei o primeiro pq é mais facil
            # print(idx, pm_idx)
            pm = self.midpoints[pm_idx]
            h = np.linalg.norm(X_bound[i, :] - pm) / 3
            u = np.sqrt(sum((x - X_bound[i,:])**2))/h
            K_total += self.kgaussian(u, h) 
        return K_total/N_train
       
    
    def predict(self, X: np.ndarray, h: float) -> np.ndarray:
        bound_elements = [item for sublist in self.bound_hyper for item in sublist]
        bound_elements = list(set(bound_elements))
        X_train = self.X_train
        y_train = self.y_train
        X_bound = self.X_train[bound_elements]
        y_bound = self.y_train[bound_elements]
        
        # Calculate P[C1], P[C2], ... , P[Cn]:
        n = np.unique(y_train).shape[0]
        K = np.zeros(n)
        p_ci =  np.zeros(n)
        for i in range(0,n):
            n_elements = np.count_nonzero(y_train==np.unique(y_train)[i])
            total_elements = y_train.shape[0]
            p_ci[i] = n_elements/total_elements
        
        # Calculate PDFs:
        y_hat = np.zeros(X.shape[0])
        index = 0; 
        for x in X:
            pdf = np.zeros(n)
            for i in range(0,n):
                indexes = np.where(y_bound==np.unique(y_bound)[i])[0]
                pdf[i] = self.my_kde(x, X_bound, indexes, h)
                K[i] = pdf[i] * p_ci[i]
            y_hat[index] = np.unique(y_train)[K.argmax()]
            index += 1
        return y_hat
