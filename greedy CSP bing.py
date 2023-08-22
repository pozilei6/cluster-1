# run in  https://www.onlinegdb.com/online_python_compiler

import numpy as np 



large_width = 400
np.set_printoptions(linewidth=large_width)



D = [7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38,
              70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56,
              88, 152, 104, 168, 200, 112, 176, 208, 224]
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



def find_approximate_solution(A, D):
    m, n = A.shape
    R = [0] * m
    C = [0] * n
    sort_C_ind = np.argsort(-A.sum(axis=0))
    for k in range(n):
        j = sort_C_ind[k]
        if A[:, j].sum() == n:
            C[j] = 0
        else:
            C[j] = 1 << k
        for i in range(m):
            if A[i, j] == 1:
                R[i] |= C[j]
                if not any([R[i] & d == R[i] for d in D]):
                    F = np.empty(shape=(m,n),dtype=bool)
                    for i in range(m):
                        for j in range(n):
                            F[i, j] = (R[i] & C[j] == C[j]) == A[i, j]
                    return k, R, C, F, np.sum(F)/(n*m)
                for jj in range(n):
                    if A[i, jj] == 1:
                        C[jj] &= R[i]
    return R, C
    
    
print(find_approximate_solution(A, D))    
start_C_index, R, C_prev, F, x = find_approximate_solution(A, D)   
sort_C_ind = np.argsort(-A.sum(axis=0))
    
def find_approximate_solution(A, D, sort_C_ind, start_C_index, C_prev):
    m, n = A.shape
    R = [0] * m
    C = [0] * n
    for k in range(start_C_index):
        C[sort_C_ind[k]] = C_prev[sort_C_ind[k]]
    for i in range(m):
        for k in range(start_C_index):
            if A[i, sort_C_ind[k]] == 1:
                R[i] |= C[sort_C_ind[k]]
    for k in range(start_C_index, n):
        j = sort_C_ind[k]
        if A[:, j].sum() == n:
            C[j] = 0
        elif all([R[i] & C[j] != C[j] for i in range(m) if A[i, j] == 0 and R[i] in D]):
            pass
        else:
            C[j] = 1 << k
        for i in range(m):
            if A[i, j] == 1:
                R[i] |= C[j]
                if not any([R[i] & d == R[i] for d in D]):
                    return None
                for jj in range(n):
                    if A[i, jj] == 1:
                        C[jj] &= R[i]
    return R, C
    
    
print(find_approximate_solution(A, D, sort_C_ind, start_C_index - 4, C_prev))   
    
# Must enforce different new value for C[start_C_index]   !!!, otherwise result is same





"""
Bing
----
Problem Formulation:

I have the following combinatorial problem. Given an instance matrix ```python A ```, which has only 0 and 1 entries (binary matrix). A has only unique rows and only unique columns. ```python D ``` is a list of integers, called the instance domain, often it's given by ```python D = [7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38,
              70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56,
              88, 152, 104, 168, 200, 112, 176, 208, 224]```. For given instance (A, D) we want to find two lists of integers ```python R ``` and ```python C ```, where len(R)=m, len(C)=n with  ```python m, n = A.shape ```. R and C have to fulfill the m*n  conditions, each is determined by entry A[i,j], i=0,...,m-1, j=0,...,n-1. They are given by two lists  ```python
 Constr_permit  = [R[i] & C[j] == C[j] for i in range(m) for j in range(n) if A[i, j] == 1]   
 Constr_npermit = [R[i] & C[j] != C[j] for i in range(m) for j in range(n) if A[i, j] == 0]   
```. So, integers have to be assigned to the m, n entries of R, C, such that  all entries of Constr_permit and all entries of Constr_npermit become True. How can we find R, C for inputs A and D?
This returns same value for all in R. So, it's not useful. The greedy approach needs to be a different one. This involves a bit of heuristics, which is, Sort range(n) indices by sum(A[:,j]) from largest to smallest and store these sorted indices in sort_C_ind. So, sort_C_ind[0] is the index j where the j'th column of A has the most 1 entries. sort_C_ind[1] the column index with second most 1 entries in the row of A of that index, and so on. With that,    


assign to C[sort_C_ind[0]]     if sum(A[:,sort_C_ind[0]])==n, then C[sort_C_ind[0]] must be assigned to 255 (set C[sort_C_ind[0]] = 255), this can't be wrong, obviously.
Recall that all columns of A are distinct, so there is no more than one column of A consisting of only 1's. So, if sum(A[:,sort_C_ind[0]]) < n, then assign an integer value
(less than 255) that has precisely one bit set (one 1 in a bit position from 0, 1, ..., 7), e.g., set sum(A[:,sort_C_ind[0]]) = 1 (=00000001 in bit vector form).
Then we need to put that set bit into all bit numbers of those R[i] that are incident with C[sort_C_ind[0]] (we say that R[i] is incident with C[j] if and only if A[i,j]==1, for
any i from 0, ... ,m-1 and j from 0, ..., n-1). This means updating ```python for i in range(m):
if A[i, sort_C_ind[0]] == 1:
R[i] = R[i] | C[sort_C_ind[0]]```, provided that we initialized ```python R= [0 for _ in range(m)]```. Then we iterate this for sort_C_ind[1], sort_C_ind[2], ..., sort_C_ind[n-1] (we call this "iteration 1").
In each iteration (at ```python R[i] = R[i] | C[sort_C_ind[0]]``` from before), we must check if R[i] can still possibly be in D, that is, is there at least one number d from D
that has all bits set where current R[i] has bits set. This can be checked by ```python any([R[i] & d == R[i] for d in D])```, I think. Once we have R[i] in D (current/intermediate R[i]
attaines a value from D), we fix it (don't change R[i] anymore for right now). Since we don't change R[i] anymore, all C entries incident with R[i] must attain all unset bits from R[i]
in their values, that is  
```python for j in range(n):
       if A[i, j] == 1:
           C[j] = C[j] & R[i]```
.
Note that, we're still within iteration 1.
This leads to a dead end at some point, as some R values are fixed but we have not ended iteration 1. So, there may be an index k from range(n) where sort_C_ind[k] isn't threated yet and A[i, sort_C_ind[k]] == 1, but
C[sort_C_ind[k]] that is infeasible. In this case (during iteration 1), we need to backtrack and assign a different set bit to a lready threated C value and continue from there with proceeding iteration 1. Can you code such
a method in a python function?
"""





# run in  https://www.onlinegdb.com/online_python_compiler

import numpy as np 
from multiprocessing import Process, Manager


large_width = 400
np.set_printoptions(linewidth=large_width)



D = [7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38,
              70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56,
              88, 152, 104, 168, 200, 112, 176, 208, 224]
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



def find_approximate_solution(A, D, best_R, best_C, best_count):
    def backtrack(A, D, sort_C_ind, start_C_index, C_prev, R, C):
        m, n = A.shape
        if start_C_index == n:
            count = sum([R[i] & C[j] == C[j] for i in range(m) for j in range(n) if A[i, j] == 1])
            if count > best_count[0]:
                best_R[:] = R
                best_C[:] = C
                best_count[0] = count
            return
        j = sort_C_ind[start_C_index]
        if A[:, j].sum() == n:
            C[j] = 0
            backtrack(A, D, sort_C_ind, start_C_index + 1, C_prev, R, C)
        else:
            for k in range(n):
                C[j] = 1 << k
                R_new = list(R)
                for i in range(m):
                    if A[i, j] == 1:
                        R_new[i] |= C[j]
                        if not any([R_new[i] & d == R_new[i] for d in D]):
                            break
                        for jj in range(n):
                            if A[i, jj] == 1:
                                C[jj] &= R_new[i]
                else:
                    backtrack(A, D, sort_C_ind, start_C_index + 1, C_prev, R_new, C)

    m, n = A.shape
    sort_C_ind = np.argsort(-A.sum(axis=0))
    R = [0] * m
    C = [0] * n
    backtrack(A, D, sort_C_ind, 0, C[:], R[:], C[:])

def find_approximate_solution_with_timeout(A, D, timeout):
    with Manager() as manager:
        best_R = manager.list([0] * A.shape[0])
        best_C = manager.list([0] * A.shape[1])
        best_count = manager.list([0])
        p = Process(target=find_approximate_solution, args=(A, D, best_R, best_C, best_count))
        p.start()
        p.join(timeout)
        p.terminate()
        return list(best_R), list(best_C)
        
        
        
        
        
timeout = 100
R, C = find_approximate_solution_with_timeout(A, D, timeout)     

print(R, C)

F = np.empty(shape=(m,n),dtype=bool)
for i in range(m):
    for j in range(n):
        F[i, j] = (R[i] & C[j] == C[j]) == A[i, j]

m,n=A.shape
print(F, np.sum(F)/(n*m))
        
        
        
        
######################################################################################################################## 
# run in  https://www.onlinegdb.com/online_python_compiler

import random
import numpy as np 

large_width = 400
np.set_printoptions(linewidth=large_width)

D = [7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38,
              70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56,
              88, 152, 104, 168, 200, 112, 176, 208, 224]
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



def fitness(R, C, A):
    m, n = A.shape
    return sum([R[i] & C[j] == C[j] for i in range(m) for j in range(n) if A[i, j] == 1])

def crossover(R1, C1, R2, C2):
    m = len(R1)
    n = len(C1)
    R = [R1[i] if random.random() < 0.5 else R2[i] for i in range(m)]
    C = [C1[j] if random.random() < 0.5 else C2[j] for j in range(n)]
    return R, C

def mutate(R, C, D):
    m = len(R)
    n = len(C)
    i = random.randrange(m)
    R[i] = random.choice(D)
    j = random.randrange(n)
    C[j] = 1 << random.randrange(8)

def find_approximate_solution(A, D):
    m, n = A.shape
    population_size = 30 #originally  100
    num_generations = 40 #originally 1000  but better lower, can run few times as res changes much
    population_R = [[random.choice(D) for _ in range(m)] for _ in range(population_size)]
    population_C = [[1 << random.randrange(8) for _ in range(n)] for _ in range(population_size)]
    for generation in range(num_generations):
        fitness_values = [fitness(population_R[i], population_C[i], A) for i in range(population_size)]
        population_R = [x for _, x in sorted(zip(fitness_values, population_R), reverse=True)]
        population_C = [x for _, x in sorted(zip(fitness_values, population_C), reverse=True)]
        new_population_R = population_R[:population_size // 2]
        new_population_C = population_C[:population_size // 2]
        while len(new_population_R) < population_size:
            i = random.randrange(population_size // 2)
            j = random.randrange(population_size // 2)
            R, C = crossover(population_R[i], population_C[i], population_R[j], population_C[j])
            mutate(R, C, D)
            new_population_R.append(R)
            new_population_C.append(C)
        population_R = new_population_R
        population_C = new_population_C
    best_R = population_R[0]
    best_C = population_C[0]
    return best_R[:], best_C[:]
    
    
R, C = find_approximate_solution(A, D)
    
print(R, C)

m,n=A.shape
F = np.empty(shape=(m,n),dtype=bool)
for i in range(m):
    for j in range(n):
        F[i, j] = (R[i] & C[j] == C[j]) == A[i, j]

print(F, np.sum(F)/(n*m))  #np.sum(F)/(n*m) = 0.546 at current population_size = 100, num_generations = 1000,          0.638 at population_size = 100, num_generations = 1000
##########################################################################################################################
        
        
        











