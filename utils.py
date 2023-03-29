import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import coo_matrix
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_curve, auc


def import_data(fisher):
    if fisher == True:
        df = pd.read_csv(r'/Users/rodrigo/Post-Grad/CC400/corr_matrices_fisher200.csv',index_col=['Institution','Subject'])
        phenotypic = pd.read_csv(r'/Users/rodrigo/Post-Grad/CC400/phenotypic200.csv',index_col=['Institution','Subject'])
    else:
        df = pd.read_csv(r'/Users/rodrigo/Post-Grad/CC400/corr_matrices200.csv',index_col=['Institution','Subject','Run'])
        phenotypic = pd.read_csv(r'/Users/rodrigo/Post-Grad/CC400/phenotypic200.csv',index_col=['Institution','Subject'])
    return df,phenotypic

def correlation_matrix(size, corr, diag=1):
    corr_matrix = np.zeros((size,size))
    cont = 0

    for i in range(size):
        for j in range(i, size):
            if i == j:
                corr_matrix[i,j] = diag #float('nan')


            else:
                corr_matrix[i,j] = corr[cont]
                corr_matrix[j,i] = corr[cont]
                cont += 1 
    return corr_matrix


def compute_KNN_graph(matrix, k_degree=10):
    """ Calculate the adjacency matrix from the connectivity matrix."""

    matrix = np.abs(matrix)
    idx = np.argsort(-matrix)[:, 0:k_degree]
    matrix.sort()
    matrix = matrix[:, ::-1]
    matrix = matrix[:, 0:k_degree]

    A = adjacency(matrix, idx).astype(np.float32)
    
    return A
    
    
def adjacency(dist, idx):

    m, k = dist.shape
    assert m, k == idx.shape
    assert dist.min() >= 0

    # Weight matrix.
    I = np.arange(0, m).repeat(k)
    J = idx.reshape(m * k)
    V = dist.reshape(m * k)
    W = coo_matrix((V, (I, J)), shape=(m, m))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    return W.todense()


def create_graph(X_train, X_test, y_train, y_test):
    
    train_data = []
    val_data = []

    # Creating train data in pyG DATA structure
    for i in range((X_train.shape[0])):
        
        # Transforming into a correlation matrix
        Adj = correlation_matrix(190,X_train.iloc[i,:].values)     
        
        # Copying the Adj matrix for operations to define edge_index
        A = Adj.copy()
        
        Adj = torch.from_numpy(Adj).float()

        # Using k-NN to define Edges
        A = compute_KNN_graph(A, 15)
                
        # Adding self connections
        np.fill_diagonal(A,1)
        A = torch.from_numpy(A).float()
        
        # getting the edge_index
        edge_index, edge_attr = dense_to_sparse(A)
        
        train_data.append(Data(x=Adj, edge_index=edge_index,edge_attr=edge_attr.reshape(len(edge_attr), 1),
                               y=torch.tensor(int(y_train.iloc[i]))))

          
    # Creating test data in pyG DATA structure
    for i in range((X_test.shape[0])):
        
        # Transforming into a correlation matrix
        Adj = correlation_matrix(190,X_test.iloc[i,:].values)     
        
        # Copying the Adj matrix for operations to define edge_index
        A = Adj.copy()
                
        Adj = torch.from_numpy(Adj).float()
        
        # Using k-NN to define Edges
        A = compute_KNN_graph(A, 15)
        
        # Adding self connections
        np.fill_diagonal(A,1)
        A = torch.from_numpy(A).float()
        
        # getting the edge_index
        edge_index, edge_attr = dense_to_sparse(A)
        

        val_data.append(Data(x=Adj, edge_index=edge_index,edge_attr=edge_attr.reshape(len(edge_attr), 1),
                             y=torch.tensor(int(y_test.iloc[i]))))

    return train_data,val_data

def create_batch(train_data, val_data, batch_size):
    
    train_loader = DataLoader(train_data, batch_size, shuffle=True)

    val_loader = DataLoader(val_data, shuffle=True)
    
    return train_loader, val_loader
    