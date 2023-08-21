##### including cliques ################## to execute in gdb-comiler  https://www.onlinegdb.com/online_python_compiler
import numpy as np
from functools import reduce
import collections
from collections import namedtuple
from copy import copy, deepcopy
from queue import Queue
    


def get_ch(j, A, X, p, H = [3, 5, 9, 6, 10, 12]):
  c = A[:,j]
  ch = reduce(lambda x,y: x & y, [H[k] for k in range(p) if any(c[X[k]])], 15)
  return ch            
def check_eq(j, j1, X, A, p, m, n, H = [3, 5, 9, 6, 10, 12]):
  cj  = A[:,j]
  cj1 = A[:,j1]
  chj  = get_ch(j, A, X, p)
  chj1 = get_ch(j1, A, X, p)
  csj, csj1 = 2, 2
  RH = [H[k] for i in range(m) for k in range(p) if i in X[k]]
  RS = [2 if cj[i] else 1 for i in range(m)]
  return [i for i in range(m) if (not cj1[i]) and (RS[i] & csj1 == csj1 and RH[i] & chj1 == chj1)]
def get_Q_eq(X, A, p, m, n, H = [3, 5, 9, 6, 10, 12]):
    Q_eq = np.zeros((n,n),dtype=int)
    for j in range(n):
        for j1 in range(n):
            if j != j1:
                Q_eq[j, j1] = len(check_eq(j, j1, X, A, p, m, n, H))
                Q_eq[j1, j] = len(check_eq(j1, j, X, A, p, m, n, H))
    return Q_eq
def get_C2_eq(Q_eq, n):
    C2_eq = []
    for j in range(n - 1):
        for j1 in range(j + 1, n):
            if Q_eq[j, j1] + Q_eq[j1, j] == 0:
                C2_eq += [(j, j1)]
    return C2_eq    
def get_C_eq(C2_eq):
  C_eq = []
  for tup in C2_eq:
    C_eq += [tup[0], tup[1]]
  return set(C_eq)

node = namedtuple("node", ['i', 'X', 'sco'])

def score(X): 
    Q_eq = get_Q_eq(X, A, p, m, n, H = [3, 5, 9, 6, 10, 12])
    C2_eq = get_C2_eq(Q_eq, n) 
    return len(C2_eq)

def appe(q, u, scob):
    for k in range(p):
        if 1 < len(u.X[k]):
            for i in u.X[k]:
                for k1 in [k1 for k1 in range(p) if k1 != k]:
                    X1 = deepcopy(u.X)
                    X1[k1].append(i)
                    X1[k].remove(i) 
                    sco = score(X1) 
                    if scob < sco: # + 1:
                        q.put(node(u.i + 1, X1, sco))   
                    if scob < sco:
                        scob = sco

def bcktr(X):
    node = namedtuple("node", ['i', 'X', 'sco'])
    q = Queue()
    best = X 
    scob = score(X)
    u = node(0, X, scob)
    q.put(u)
    while not q.empty():
        u = q.get()
        print(u.i, u.sco)
        if scob < u.sco:
            scob = u.sco
            best = u.X
        if 4 < u.i or 38 < scob:
            return best, scob
        if scob < u.sco + 1:
            appe(q, u, scob)
    return best, scob




#instance 
#n_clusters = 6
#X = [list(np.array(sorted(c),dtype=int)) for c in louvain_method(np.matmul(A,A.transpose()),n_clusters)[0]] #initial by louvain, replaces below X
X = [[21,19,16,17], [18,4,7,23,14], [22,3,11,20,9,1], [13], [8,5,2,0,15], [6,12,10]]

A = np.array(
    [#1 2 3 4 5 6 7 8 9 10111213141516171819
     [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1], # 1   d
     [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # 2  d
     [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0], # 3   d
     [0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0], # 4  d
     [1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0], # 5 d
     [0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0], # 6   d
     [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0], # 7     d
     [1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0], # 8 d
     [0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,0,0], # 9   d
     [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0], # 10  d
     [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0], # 11    d
     [0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0], # 12  d
     [1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0], # 13    d
     [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0], # 14    d
     [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1], # 15 d
     [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0], # 16   d
     [0,0,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,0,0], # 17 d
     [0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0], # 18 d
     [1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0], # 19 d
     [0,0,0,0,0,1,1,0,1,0,0,0,0,1,1,0,0,0,0], # 20 d
     [0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], # 21  d
     [0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,0,0], # 22 d
     [0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,1], # 23  d
     [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0], # 24 d
     ],dtype = int)

m, n = A.shape
p = len(X)

best, scob = bcktr(X)
print(best)
for b in best:
    for i in b:
        print(A[i], i)
    print()

Q_eq = get_Q_eq(best, A, p, m, n, H = [3, 5, 9, 6, 10, 12])
C2_eq = get_C2_eq(Q_eq, n) 

#get cliques 
def get_adj(C2_eq, n):
    M = np.zeros((n, n), dtype=int)
    for (j, j1) in C2_eq:
        M[j, j1], M[j1, j] = 1, 1
    adj = {j: [] for j in range(n)}
    for j in range(n):
        adj[j] = [j1 for j1 in range(n) if M[j, j1]]
    return adj 

adj = get_adj(C2_eq, n)

def cliques_recursive(neighbors, r, p, x):
    if not p and not x:
        yield r
    else:
        for v in min((p - neighbors[u] for u in p | x), key=len):
            yield from cliques_recursive(
                neighbors, r | {v}, p & neighbors[v], x & neighbors[v]
            )
            p.remove(v)
            x.add(v)


def cliques(graph):
    neighbors = collections.defaultdict(set)
    for v, n_v in graph.items():
        for u in n_v:
            if u != v:
                neighbors[u].add(v)
                neighbors[v].add(u)
    yield from cliques_recursive(neighbors, set(), set(neighbors), set())

print("cliques")
for clique in cliques(adj):
    print(clique)


for b in best:
    for i in b:
        print([j+1 for j in range(n) if A[i,j]], i)
    print()

print(C2_eq)
print(len(C2_eq))

Q_eq = get_Q_eq(X, A, p, m, n, H = [3, 5, 9, 6, 10, 12])
C2_eq = get_C2_eq(Q_eq, n) 

print(C2_eq)
print(len(C2_eq))
