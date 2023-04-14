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


def reconstruct_symmetric_matrix(size, upper_triangle_array, diag=1):

    result = np.zeros((size, size))
    result[np.triu_indices_from(result, 1)] = upper_triangle_array
    result = result + result.T
    np.fill_diagonal(result, diag)
    return result


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


def create_graph(X_train, X_test, y_train, y_test, method={'knn' : 10}):
    
    train_data = []
    val_data = []

    # Creating train data in pyG DATA structure
    for i in range((X_train.shape[0])):
        
        # Transforming into a correlation matrix
        Adj = reconstruct_symmetric_matrix(190,X_train.iloc[i,:].values)     
        
        # Copying the Adj matrix for operations to define edge_index
        A = Adj.copy()
        
        Adj = torch.from_numpy(Adj).float()
        
        if method == None:
            A = A

        elif list(method.keys())[0] =='knn':
            # Using k-NN to define Edges
            A = compute_KNN_graph(A, 10)

        elif list(method.keys())[0] =='threshold':
            A[A < method['threshold']] = 0
            Adj[Adj < method['threshold']] = 0


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
        Adj = reconstruct_symmetric_matrix(190,X_test.iloc[i,:].values)     
        
        # Copying the Adj matrix for operations to define edge_index
        A = Adj.copy()
                
        Adj = torch.from_numpy(Adj).float()
        
        if method == None:
            A = A   
        
        elif list(method.keys())[0] =='knn':
            # Using k-NN to define Edges
            A = compute_KNN_graph(A, 10)
            
        elif list(method.keys())[0] =='threshold':
            A[A < method['threshold']] = 0
            Adj[Adj < method['threshold']] = 0
          
        
        # Adding self connections
        np.fill_diagonal(A,1)
        A = torch.from_numpy(A).float()
        
        # getting the edge_index
        edge_index, edge_attr = dense_to_sparse(A)
        

        val_data.append(Data(x=Adj, edge_index=edge_index,edge_attr=edge_attr.reshape(len(edge_attr), 1),
                             y=torch.tensor(int(y_test.iloc[i]))))

    return train_data,val_data

def create_batch(train_data, val_data, batch_size):
    
    train_loader = DataLoader(train_data, batch_size) #Shuffle=True

    val_loader = DataLoader(val_data)  # Shuffle=True
    
    return train_loader, val_loader
    