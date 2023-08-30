
#--------------------train pruning---------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Define a function to extract features from an instance (A, D, R)
def extract_features(A, D, R):
    # Compute the size of the instance
    m, n = A.shape
    # Compute the number of non-zero elements in A
    nnz = np.count_nonzero(A)
    # Compute the average value of D
    avg_D = np.mean(D)
    # Compute the number of set values in R
    n_set_R = sum([1 for r in R if r is not None])
    # Flatten A and R and concatenate them with D and the additional features
    features = np.concatenate((A.flatten(), D, R, [nnz, avg_D, n_set_R]))
    # Return the features as a numpy array
    return np.array(features)

# Define a function to generate training data from a list of instances (A, D, R)
def generate_training_data(instances):
    X = []
    y = []
    for A, D, R, label in instances:
        # Extract the features from the instance (A, D, R)
        x = extract_features(A, D, R)
        # Append the features and label to the training data
        X.append(x)
        y.append(label)
    return np.array(X), np.array(y)

# Define a list of instances (A, D, R) with known labels
instances = [
    (np.array([[1, 0], [0, 1]]), [1, 2], [None, None], 1),
    (np.array([[1, 1], [1, 1]]), [1, 2], [None, None], 0),
    # ...
]

# Generate the training data from the instances
X_train, y_train = generate_training_data(instances)

# Create a decision tree classifier with cost complexity pruning
clf = DecisionTreeClassifier(ccp_alpha=0.1)

# Train the classifier on the training data
clf.fit(X_train, y_train)

def solve_csp(A, D):
    m, n = A.shape
    R = [None] * m

    def is_valid(R, r, v):
        R[r] = v
        for j in range(n):
            C = reduce(lambda x,y: x & y,[R[i] for i in range(m) if A[i,j] == 1 and R[i] is not None],2**24-1)
            for i in range(r+1):
                if (R[i] & C == C) != A[i,j]:
                    R[r] = None
                    return False
        R[r] = None
        return True

    def dfs(r):
        if r == m:
            return R[:]
        for v in D:
            if is_valid(R,r,v):
                x = extract_features(A,D,R)
                y_pred = clf.predict([x])
                if y_pred == 0:
                    continue
                R[r] = v
                result = dfs(r+1)
                if result is not None:
                    return result
                R[r] = None
        return None

    return dfs(0)

# Test the solve_csp function on an instance (A_test,D_test)
A_test = np.array([[1,0],[0,1]])
D_test = [1,2]
R_test = solve_csp(A_test,D_test)
print(R_test)
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import SpectralClustering
from sknetwork.clustering import Louvain, Leiden

def seriate(A):
    # Compute the Jaccard distance between rows of A
    Y = pdist(A, metric='jaccard')
    # Compute the optimal leaf ordering of a hierarchical clustering
    Z = linkage(Y, method='ward')
    ser_I = optimal_leaf_ordering(Z, Y)
    # Return the seriated matrix A_ser
    return A[ser_I,:]

def agglomerative_clustering(A, n_clusters):
    # Compute the Jaccard distance between rows of A
    Y = pdist(A, metric='jaccard')
    # Perform agglomerative clustering with Ward's method
    Z = linkage(Y, method='ward')
    # Form flat clusters from the hierarchical clustering
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    # Return the clusters as a list of lists of row indices
    X = [np.where(clusters == i)[0].tolist() for i in range(1,n_clusters+1)]
    return X

def community_detection(A, method):
    # Compute the adjacency matrix of the bipartite graph
    B = A.dot(A.T)
    np.fill_diagonal(B, 0)
    # Perform community detection using Louvain or Leiden method
    if method == 'louvain':
        louvain = Louvain()
        clusters = louvain.fit_transform(B)
    elif method == 'leiden':
        leiden = Leiden()
        clusters = leiden.fit_transform(B)
    else:
        raise ValueError('Invalid method: {}'.format(method))
    # Return the clusters as a list of lists of row indices
    n_clusters = np.unique(clusters).size
    X = [np.where(clusters == i)[0].tolist() for i in range(n_clusters)]
    return X

def spectral_clustering(A, n_clusters):
    # Compute the Jaccard distance between rows of A
    Y = pdist(A, metric='jaccard')
    # Perform spectral clustering on the distance matrix
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    clusters = sc.fit_predict(squareform(Y))
    # Return the clusters as a list of lists of row indices
    X = [np.where(clusters == i)[0].tolist() for i in range(n_clusters)]
    return X

def extract_features(A_ser_shape,A_ser,D,R,X_aggl,X_comm,X_spec):
# Compute size of instance from A_ser_shape instead of A_ser.shape.
m,n = A_ser_shape
# Compute number of non-zero elements in A_ser.
nnz = np.count_nonzero(A_ser)
# Compute average value of D.
avg_D = np.mean(D)
# Compute number of set values in R.
n_set_R = sum([1 for r in R if r is not None])
# Flatten A_ser and R and concatenate them with D and other features.
# Convert clusterings to arrays with fixed size.
# Flatten clusterings and concatenate them with other features.
# Return features as numpy array.
features = np.concatenate((A_ser.flatten(),D,R,
[np.array(X_aggl).flatten(),np.array(X_comm).flatten(),
np.array(X_spec).flatten(),nnz,avg_D,n_set_R]))
return np.array(features)

# Define function to generate training data from list of instances (A,D,R).
def generate_training_data(instances,n_clusters_aggl,n_clusters_spec,
method_comm):
X = []
y = []
for A,A_shape,D,n_clusters_aggl,n_clusters_spec,label in instances:
# Compute seriated matrix A_ser.
A_ser = seriate(A)
# Perform agglomerative clustering on A_ser.
X_aggl = agglomerative_clustering(A_ser,n_clusters_aggl)
# Perform community detection on A_ser.
X_comm = community_detection(A_ser,method_comm)
# Perform spectral clustering on A_ser.
X_spec = spectral_clustering(A_ser,n_clusters_spec)
# Extract features from instance (A_ser_shape,A_ser,D,R,X_aggl,X_comm,X_spec).
x = extract_features(A_shape,A_ser,D,R,X_aggl,X_comm,X_spec)
# Append features and label to training data.
X.append(x)
y.append(label)
return np.array(X),np.array(y)

# Define list of instances (A,D,R) with known labels.
instances = [
(np.array([[1,0],[0,1]]),(2,2),[1,2],2,2,1),
(np.array([[1,1],[1,1]]),(2,2),[1,2],2,2,0),
# ...
]

# Set parameters for clustering.
n_clusters_aggl = 2
n_clusters_spec = 2
method_comm = 'louvain'

# Generate training data from instances.
X_train,y_train = generate_training_data(instances,n_clusters_aggl,
n_clusters_spec,method_comm)

# Create decision tree classifier with cost complexity pruning.
clf = DecisionTreeClassifier(ccp_alpha=0.1)

# Train classifier on training data.
clf.fit(X_train,y_train)

def solve_csp(A,D):
m,n = A.shape
R = [None] * m

def is_valid(R,r,v):
R[r] = v
for j in range(n):
C = reduce(lambda x,y: x & y,[R[i] for i in range(m) if A[i,j] == 1 and R[i] is not None],2**24-1)
for i in range(r+1):
if (R[i] & C == C) != A[i,j]:
R[r] = None
return False
R[r] = None
return True

def dfs(r):
if r == m:
return R[:]
for v in D:
if is_valid(R,r,v):
x = extract_features((m,n),A,D,R,X_aggl,X_comm,X_spec)
y_pred = clf.predict([x])
if y_pred == 0:
continue
R[r] = v
result = dfs(r+1)
if result is not None:
return result
R[r] = None
return None

# Compute seriated matrix A_ser and clusterings X_aggl, X_comm, X_spec.
A_ser = seriate(A)
X_aggl = agglomerative_clustering(A_ser,n_clusters_aggl)
X_comm = community_detection(A_ser,method_comm)
X_spec = spectral_clustering(A_ser,n_clusters_spec)

return dfs(0)

# Test solve_csp function on instance (A_test,D_test).
A_test = np.array([[1,0],[0,1]])
D_test = [1,2]
R_test = solve_csp(A_test,D_test)
print(R_test)
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

























