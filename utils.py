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
from neurocombat_sklearn import CombatModel

np.random.seed(42)



def import_data(fisher):
    if fisher == True:
        df = pd.read_csv(r'/Users/rodrigo/Post-Grad/CC400/corr_matrices_fisher200.csv',index_col=['Institution','Subject'])
        phenotypic = pd.read_csv(r'/Users/rodrigo/Post-Grad/CC400/phenotypic200.csv',index_col=['Institution','Subject'])
    else:
        #df = pd.read_csv(r'/Users/rodrigo/Post-Grad/CC400/corr_matrices200_half.csv',index_col=['Institution','Subject','Run'])
        df = pd.read_csv(r'/Users/rodrigo/Post-Grad/CC400/corr_matrices200.csv',index_col=['Institution','Subject','Run'])
        phenotypic = pd.read_csv(r'/Users/rodrigo/Post-Grad/CC400/phenotypic200.csv',index_col=['Institution','Subject'])
    return df,phenotypic

def remove_triangle(df):
    # Remove triangle of a symmetric matrix and the diagonal
    
    df = df.astype(float)
    df.values[np.triu_indices_from(df, k=1)] = np.nan
    df  = ((df.T).values.reshape((1,(df.shape[0])**2)))
    df = df[~np.isnan(df)]
    df = df[df!=1]
    return (df).reshape((1,len(df)))


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


def create_graph(X_train, X_test, y_train, y_test, size=190 ,method={'knn' : 10}):
    
    train_data = []
    val_data = []

    # Creating train data in pyG DATA structure
    for i in range((X_train.shape[0])):
        
        # Transforming into a correlation matrix
        Adj = reconstruct_symmetric_matrix(size,X_train.iloc[i,:].values)     
        
        # Copying the Adj matrix for operations to define edge_index
        A = Adj.copy()
        
        Adj = torch.from_numpy(Adj).float()
        
        if method == None:
            A = A

        elif list(method.keys())[0] =='knn':
            # Using k-NN to define Edges
            A = compute_KNN_graph(A, method['knn'])

        elif list(method.keys())[0] =='threshold':
            A[A < method['threshold']] = 0
            Adj[Adj < method['threshold']] = 0
            
        elif list(method.keys())[0] == 'knn_group':
            A = method['knn_group']


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
        Adj = reconstruct_symmetric_matrix(size,X_test.iloc[i,:].values)     
        
        # Copying the Adj matrix for operations to define edge_index
        A = Adj.copy()
                
        Adj = torch.from_numpy(Adj).float()
        
        if method == None:
            A = A   
        
        elif list(method.keys())[0] =='knn':
            # Using k-NN to define Edges
            A = compute_KNN_graph(A, method['knn'])
            
        elif list(method.keys())[0] =='threshold':
            A[A < method['threshold']] = 0
            Adj[Adj < method['threshold']] = 0
            
        elif list(method.keys())[0] == 'knn_group':
            A = method['knn_group']
          
        
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
    
def cross_val_data(df, folds=10, site=True):
    X_train_final = []
    y_train_final = []
    X_test_final = []
    y_test_final = []
    
    arr = df['Subject'].unique()
    #np.random.shuffle(arr)
    kfold = np.array_split(arr, folds)
    
    for i in range(folds):
        test_loss_hist = 0

        df_train = df[~df.Subject.isin(kfold[i])]
        df_test = df[df.Subject.isin(kfold[i])]


        Site_train = df_train[['Site']]
        X_train = df_train.drop(columns=['Institution', 'Subject', 'Run','Gender', 'Age', 'Site'])#,'Half'])
        y_train = df_train.Gender

        Site_test = df_test[['Site']]
        X_test = df_test.drop(columns=['Institution', 'Subject', 'Run', 'Gender', 'Age', 'Site'])#, 'Half'])
        y_test = df_test.Gender
        
        y_train_final.append(y_train)
        y_test_final.append(y_test)
        
        if site == True:

            # Creating model
            combat = CombatModel()

            # Fitting the model and transforming the training set
            X_train_final.append(combat.fit_transform(X_train.values,
                                                     Site_train)) #X_train_har

            # Harmonize test set using training set fitted parameters
            X_test_final.append(combat.transform(X_test.values,
                                                Site_test)) #X_test_har
            
        else:
            X_train_final.append(X_train.values)
            X_test_final.append(X_test.values)
        
    return X_train_final, y_train_final, X_test_final, y_test_final
