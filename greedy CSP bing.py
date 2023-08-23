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
from functools import reduce

large_width = 400
np.set_printoptions(linewidth=large_width)


def concat_two_codes(codes2,codes1,bit_len_2,bit_len_1):
    shifted_codes2 = [c2 << bit_len_1 for c2 in codes2]
    codes2_con_codes1 = [sh_c2 + c1 for sh_c2 in shifted_codes2 for c1 in codes1]    
    return codes2_con_codes1

D = [7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38,
              70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56,
              88, 152, 104, 168, 200, 112, 176, 208, 224]
D = concat_two_codes(D,D,8,8)
H = [7, 11, 13, 14]
#D = concat_two_codes(H,D,4,16)
A = np.array(
    [#1 2 3 4 5 6 7 8 9 10111213141516171819
     [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1], # 1   d
     [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # 2  d
     [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0], # 3   d
     [0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0], # 4  d
     [1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0], # 5 d
     [0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,1,0,0], # 6   d
     [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0], # 7     d
     [1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0], # 8 d
     [0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,0,0], # 9   d
     [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0], # 10  d
     [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0], # 11    d
     [0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0], # 12  d
     [1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0], # 13    d
     [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0], # 14    d
     [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1], # 15 d
     ],dtype = int)
"""
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
"""


#-------------------------------------------------------------------------------
def fitness(R, A):
    m, n = A.shape
    defa = 2**16 - 1
    C = [reduce(lambda x,y: x & y, [R[i] for i in range(m) if A[i, j] == 1], defa) for j in range(n)]
    return sum([(R[i] & C[j] == C[j]) == A[i, j] for i in range(m) for j in range(n)])

def crossover(R1, R2):
    m = len(R1)
    R = [R1[i] if random.random() < 0.5 else R2[i] for i in range(m)]
    return R

def mutate(R, D):
    m = len(R)
    i = random.randrange(m)
    R[i] = random.choice(D)
    
def find_approximate_solution(A, D, pop_size=30, num_genera=400):
    m, n = A.shape
    max_fitness = m * n
    population_size = pop_size
    num_generations = num_genera
    population_R = [[random.choice(D) for _ in range(m)] for _ in range(population_size)]
    for generation in range(num_generations):
        fitness_values = [fitness(population_R[i], A) for i in range(population_size)]
        # Sort the population by fitness
        population_R = [x for _, x in sorted(zip(fitness_values, population_R), reverse=True)]
        # If the fittest individual has maximum fitness, return it immediately
        if fitness_values[0] == max_fitness:
            return population_R[0]
        # Otherwise, generate the next generation
        new_population_R = population_R[:population_size // 2]
        while len(new_population_R) < population_size:
            i = random.randrange(population_size // 2)
            j = random.randrange(population_size // 2)
            R = crossover(population_R[i], population_R[j])
            mutate(R, D)
            new_population_R.append(R)
        population_R = new_population_R
    # If no individual with maximum fitness was found, return the fittest individual from the last generation
    return population_R[0]
#-------------------------------------------------------------------------------

def split_bin(B, b_le_right):
    r_1s = 2**b_le_right - 1
    B_right = [b & r_1s for b in B]
    B_left  = [b >> b_le_right for b in B]
    return B_left, B_right
    
def is_SAT(m, n, R, C):
    F = np.empty(shape=(m,n),dtype=bool)
    for i in range(m):
        for j in range(n):
            F[i, j] = (R[i] & C[j] == C[j]) == A[i, j]  
    print(np.sum(F), n * m)
    return np.sum(F) == n * m  # SAT?

def genetic_solver(A, D, H, b_le_S, b_le_H):
    defa = 2**(b_le_S + b_le_H) - 1
    m, n = A.shape
    D = concat_two_codes(H, D, b_le_H, b_le_S)
    pop_size = (m + n) // 2 
    num_genera = m * n 
    for v in range(20):
        R = find_approximate_solution(A, D, pop_size + 2 * v * m, num_genera + v * (m + n))
        C = [reduce(lambda x,y: x & y, [R[i] for i in range(m) if A[i, j] == 1], defa) for j in range(n)]
        
        if is_SAT(m, n, R, C):                          
            RH_int, RS_int = split_bin(R, b_le_S)
            CH_int, CS_int = split_bin(C, b_le_S)
            R_int = [RH_int, RS_int]
            C_int = [CH_int, CS_int]
            return [R_int, C_int]     
            
    return np.sum(F)/(n*m)          # UNSAT
    
        
        
R_int, C_int = genetic_solver(A, D, H, 16, 4)

##########################################################################################################################
        
        
        

########################################################################################################################### Artificial Bee Colony (ABC) algorithm ###########################
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
    return sum([(R[i] & C[j] == C[j]) == A[i, j] for i in range(m) for j in range(n)])

def generate_solution(A, D):
    m, n = A.shape
    R = [random.choice(D) for _ in range(m)]
    C = [1 << random.randrange(8) for _ in range(n)]
    return R, C

def mutate_solution(R, C, D):
    m = len(R)
    n = len(C)
    i = random.randrange(m)
    R[i] = random.choice(D)
    j = random.randrange(n)
    C[j] = 1 << random.randrange(8)

def find_approximate_solution(A, D):
    m, n = A.shape
    population_size = 2
    num_iterations = 40
    population_R = []
    population_C = []
    for _ in range(population_size):
        R, C = generate_solution(A, D)
        population_R.append(R)
        population_C.append(C)
    fitness_values = [fitness(population_R[i], population_C[i], A) for i in range(population_size)]
    for iteration in range(num_iterations):
        for i in range(population_size):
            R_new = list(population_R[i])
            C_new = list(population_C[i])
            mutate_solution(R_new, C_new, D)
            fitness_new = fitness(R_new, C_new, A)
            if fitness_new > fitness_values[i]:
                population_R[i] = R_new
                population_C[i] = C_new
                fitness_values[i] = fitness_new
        best_index = max(range(population_size), key=lambda i: fitness_values[i])
        for i in range(population_size):
            if i != best_index:
                R_new = list(population_R[best_index])
                C_new = list(population_C[best_index])
                mutate_solution(R_new, C_new, D)
                fitness_new = fitness(R_new, C_new, A)
                if fitness_new > fitness_values[i]:
                    population_R[i] = R_new
                    population_C[i] = C_new
                    fitness_values[i] = fitness_new
    best_index = max(range(population_size), key=lambda i: fitness_values[i])
    best_R = population_R[best_index]
    best_C = population_C[best_index]
    return best_R[:], best_C[:]


#execution
R, C = find_approximate_solution(A, D)
    
print(R, C)

m,n=A.shape
F = np.empty(shape=(m,n),dtype=bool)
for i in range(m):
    for j in range(n):
        F[i, j] = (R[i] & C[j] == C[j]) == A[i, j]

print(F, np.sum(F)/(n*m))    
    
print(all(r in set(D) for r in R))    #r all in D
for r in R:
    print(np.binary_repr(r, 8))    

########################################################################################################################### Artificial Bee Colony (ABC) algorithm ###########################



#############################################################################################################################################################################################
import numpy as np
from functools import reduce


def concat_two_codes(codes2,codes1,bit_len_2,bit_len_1):
    shifted_codes2 = [c2 << bit_len_1 for c2 in codes2]
    codes2_con_codes1 = [sh_c2 + c1 for sh_c2 in shifted_codes2 for c1 in codes1]    
    return codes2_con_codes1

D = [7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38,
              70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56,
              88, 152, 104, 168, 200, 112, 176, 208, 224]
D = concat_two_codes(D,D,8,8)
H = [7, 11, 13, 14]
#D = concat_two_codes(H,D,4,16)
A = np.array(
    [#1 2 3 4 5 6 7 8 9 10111213141516171819
     [1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1], # 1   d
     [0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0], # 2  d
     [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0], # 3   d
     [0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0], # 4  d
     [1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0], # 5 d
     [0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,1,0,0], # 6   d
     [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0], # 7     d
     [1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0], # 8 d
     [0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,0,0], # 9   d
     [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0], # 10  d
     [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0], # 11    d
     [0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0], # 12  d
     [1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0], # 13    d
     [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0], # 14    d
     [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1], # 15 d
     ],dtype = int)
     
import math
import random

def fitness(R, A):
    m, n = A.shape
    defa = 2**20 - 1
    C = [reduce(lambda x,y: x & y, [R[i] for i in range(m) if A[i, j] == 1], defa) for j in range(n)]
    return sum([(R[i] & C[j] == C[j]) == A[i, j] for i in range(m) for j in range(n)])
    
def mutate(R, D):
    m = len(R)
    i = random.randrange(m)
    R[i] = random.choice(D)

def simulated_annealing(A, D, max_iterations=1000, initial_temperature=100, cooling_rate=0.95):
    m, n = A.shape
    max_fitness = m * n
    defa = 2**20 - 1
    
    # Initialize the current solution and its fitness
    current_solution = [random.choice(D) for _ in range(m)]
    current_fitness = fitness(current_solution, A)
    
    # Set the initial temperature and the best solution found so far
    temperature = initial_temperature
    best_solution = current_solution
    best_fitness = current_fitness
    
    for i in range(max_iterations):
        # Create a new solution by mutating the current solution
        new_solution = current_solution[:]
        mutate(new_solution, D)
        new_fitness = fitness(new_solution, A)
        
        # Calculate the change in fitness
        delta_fitness = new_fitness - current_fitness
        
        # If the new solution is better, accept it
        if delta_fitness > 0:
            current_solution = new_solution
            current_fitness = new_fitness
            
            # Update the best solution found so far
            if new_fitness > best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                
                # If the best solution has maximum fitness, return it immediately
                if best_fitness == max_fitness:
                    return best_solution
                
        # If the new solution is worse, accept it with a certain probability
        else:
            acceptance_probability = math.exp(delta_fitness / temperature)
            if random.random() < acceptance_probability:
                current_solution = new_solution
                current_fitness = new_fitness
        
        # Cool down the temperature
        temperature *= cooling_rate
    
    return best_solution

# Example usage:
m, n = A.shape
defa = 2**(16 + 4) - 1
R = simulated_annealing(A, D)
C = [reduce(lambda x,y: x & y, [R[i] for i in range(m) if A[i, j] == 1], defa) for j in range(n)]


def split_bin(B, b_le_right):
    r_1s = 2**b_le_right - 1
    B_right = [b & r_1s for b in B]
    B_left  = [b >> b_le_right for b in B]
    return B_left, B_right
    
def is_SAT(m, n, R, C):
    F = np.empty(shape=(m,n),dtype=bool)
    for i in range(m):
        for j in range(n):
            F[i, j] = (R[i] & C[j] == C[j]) == A[i, j]  
    print(np.sum(F), n * m)
    return np.sum(F) == n * m  # SAT?

def genetic_solver(A, D, H, b_le_S, b_le_H):
    
    defa = 2**(b_le_S + b_le_H) - 1
    m, n = A.shape
    D = concat_two_codes(H, D, b_le_H, b_le_S)

    for v in range(20):
        R = simulated_annealing(A, D, max_iterations=2000, initial_temperature=300, cooling_rate=0.60)
        C = [reduce(lambda x,y: x & y, [R[i] for i in range(m) if A[i, j] == 1], defa) for j in range(n)]
        
        if is_SAT(m, n, R, C):                          
            RH_int, RS_int = split_bin(R, b_le_S)
            CH_int, CS_int = split_bin(C, b_le_S)
            R_int = [RH_int, RS_int]
            C_int = [CH_int, CS_int]
            return [R_int, C_int]     
            
    return 'UNSAT'

R_int, C_int = genetic_solver(A, D, H, 16, 4)
#############################################################################################################################################################################################






############################################################################ simulated annealing 2 ##############################################################################################
import numpy as np
from functools import reduce
from itertools import combinations

def get_H(le, nbr_0s):
    # Get all possible combinations of positions for the unset bits
    unset_bits_positions = list(combinations(range(le), nbr_0s))
    H = []
    # Set all bits to 1
    num = (1 << le) - 1
    for positions in unset_bits_positions:
        # Unset the bits at the specified positions
        temp_num = num
        for pos in positions:
            temp_num &= ~(1 << pos)
        H.append(temp_num)
    return H

def concat_two_codes(codes2,codes1,bit_len_2,bit_len_1):
    shifted_codes2 = [c2 << bit_len_1 for c2 in codes2]
    codes2_con_codes1 = [sh_c2 + c1 for sh_c2 in shifted_codes2 for c1 in codes1]    
    return codes2_con_codes1
    
    

D = [7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38,
              70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56,
              88, 152, 104, 168, 200, 112, 176, 208, 224]
D = concat_two_codes(D,D,8,8)
H = [7, 11, 13, 14]
H = get_H(7, 1)
#D = concat_two_codes(H,D,4,16)
A = np.array(
    [#1 2 3 4 5 6 7 8 9 10111213141516171819
     [1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1], # 1   d
     [0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0], # 2  d
     [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0], # 3   d
     [0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0], # 4  d
     [1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0], # 5 d
     [0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,1,0,0], # 6   d
     [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0], # 7     d
     [1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0], # 8 d
     [0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,0,0], # 9   d
     [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0], # 10  d
     [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0], # 11    d
     [0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0], # 12  d
     [1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0], # 13    d
     [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0], # 14    d
     [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1], # 15 d
     ],dtype = int)
     
import math
import random

def highest_unset_bit(bit_le, R):
    highest_position = -1
    for num in R:
        for pos in range(bit_le):
            if (num & (1 << pos)) == 0:
                highest_position = max(highest_position, pos)
    return highest_position
    
def current_sol(D, A, m):
    n = A.shape[1]
    current_solution = []
    while len(current_solution) < m:
        # Choose a random number from D
        num = random.choice(D)
        # Check if the number satisfies the properties
        valid = True
        for i in range(len(current_solution)):
            if num == current_solution[i]:
                valid = False
                break
            if any(A[i, k] == 1 and A[len(current_solution), k] == 1 for k in range(n)):
                if (num & current_solution[i]) == 0:
                    valid = False
                    break
        if valid:
            current_solution.append(num)
    return current_solution

def fitness(R, A, b_le_S, b_le_H):
    m, n = A.shape
    defa = 2**21 - 1
    C = [reduce(lambda x,y: x & y, [R[i] for i in range(m) if A[i, j] == 1], defa) for j in range(n)]
    return sum([(R[i] & C[j] == C[j]) == A[i, j] for i in range(m) for j in range(n)]) * highest_unset_bit(b_le_S + b_le_H, R)
    
def mutate(R, D):
    m = len(R)
    i = random.randrange(m)
    R[i] = random.choice(D)





#------------------------------------
def simulated_annealing_with_restarts(A, D, b_le_S, b_le_H, num_restarts=10, max_iterations=1000, initial_temperature=100, cooling_rate=0.95):
    m, n = A.shape
    max_fitness = m * n
    defa = 2**21 - 1
    
    best_solution = None
    best_fitness = 0
    
    for restart in range(num_restarts):
        # Initialize the current solution and its fitness
        #current_solution = random.sample(D, m)
        current_solution = current_sol(D, A, m)
        #current_solution = [random.choice(D) for _ in range(m)]
        current_fitness = fitness(current_solution, A, b_le_S, b_le_H)
        
        # Set the initial temperature
        temperature = initial_temperature
        
        for i in range(max_iterations):
            # Create a new solution by mutating the current solution
            new_solution = current_solution[:]
            mutate(new_solution, D)
            new_fitness = fitness(new_solution, A, b_le_S, b_le_H)
            
            # Calculate the change in fitness
            delta_fitness = new_fitness - current_fitness
            
            # If the new solution is better, accept it
            if delta_fitness > 0:
                current_solution = new_solution
                current_fitness = new_fitness
                
                # Update the best solution found so far
                if new_fitness > best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
                    
                    # If the best solution has maximum fitness, return it immediately
                    if best_fitness == max_fitness:
                        return best_solution
                    
            # If the new solution is worse, accept it with a certain probability
            else:
                if np.isclose(temperature, 0):
                    acceptance_probability = 0
                else:
                    acceptance_probability = np.exp(delta_fitness / temperature)
                if random.random() < acceptance_probability:
                    current_solution = new_solution
                    current_fitness = new_fitness
            
            # Cool down the temperature
            temperature *= cooling_rate
    
    return best_solution
#------------------------------------




# Example usage:
m, n = A.shape
defa = 2**(16 + 4) - 1



def split_bin(B, b_le_right):
    r_1s = 2**b_le_right - 1
    B_right = [b & r_1s for b in B]
    B_left  = [b >> b_le_right for b in B]
    return B_left, B_right
    
def is_SAT(m, n, R, C):
    F = np.empty(shape=(m,n),dtype=bool)
    for i in range(m):
        for j in range(n):
            F[i, j] = (R[i] & C[j] == C[j]) == A[i, j]  
    print(np.sum(F), n * m)
    return np.sum(F) == n * m  # SAT?

def genetic_solver(A, D, H, b_le_S, b_le_H):
    
    defa = 2**(b_le_S + b_le_H) - 1
    m, n = A.shape
    D = concat_two_codes(H, D, b_le_H, b_le_S)

    for v in range(20):
        R = simulated_annealing_with_restarts(A, D, b_le_S, b_le_H, num_restarts=30, max_iterations=1000, initial_temperature=200, cooling_rate=0.99)
        C = [reduce(lambda x,y: x & y, [R[i] for i in range(m) if A[i, j] == 1], defa) for j in range(n)]
        
        if is_SAT(m, n, R, C):                          
            RH_int, RS_int = split_bin(R, b_le_S)
            CH_int, CS_int = split_bin(C, b_le_S)
            R_int = [RH_int, RS_int]
            C_int = [CH_int, CS_int]
            return [R_int, C_int]     
            
    return 'UNSAT'


print(current_sol(D, A, m))


H = get_H(3, 1)
R_int, C_int = genetic_solver(A, D, H, 16, 3)



############################################################################ simulated annealing 2 ##############################################################################################








########## plotting by (population size, number of iterations) #############################################


"""
Quesion:
I want to explore the effect of changing population_size and num_generations settings settings in your latest find_approximate_solution(A, D) (the one using ABC) (so, it's then find_approximate_solution(A, D, population_size, num_generations)).
Can you add these two to parameter list. Write code that, for fixed instance A and D, iterates through population_size from 1 to 50, in steps of 5, and in inner loop num_generations from 1 to 60, in steps of 5, 
and 3D-plots a measure of approximation ```python me```, computed by 
```python
R, C = find_approximate_solution(A, D, population_size, num_generations)
m, n = A.shape
F = np.empty(shape=(m,n),dtype=bool)
for i in range(m):
    for j in range(n):
        F[i, j] = (R[i] & C[j] == C[j]) == A[i, j]
me = np.sum(F)/(n*m)
```.
So independent axes x, y are population_size, num_generations and dependent (resulting) axis is me. Show me the code with neccessary packages, e.g., sklearn, imported.
"""

# run in Colab

import numpy as np 

D = [7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38,
              70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56,
              88, 152, 104, 168, 200, 112, 176, 208, 224]
A = np.array(
    [#1 2 3 4 5 6 7 8 9 10111213141516171819
     [1,1,0,0,0,0,0], # 1   d
     [1,0,1,0,1,0,0], # 2  d
     [1,0,0,1,0,0,0], # 3   d
     [1,0,0,0,1,0,0], # 4  d
     [1,0,0,0,0,0,0], # 5 d
     [0,0,0,1,0,1,0], # 6   d
     [0,0,0,0,0,1,1], # 7     d
     [0,0,0,1,0,1,0],
     [0,1,0,0,0,0,1],
     [1,0,0,0,0,1,1],
     ],dtype = int)

import random
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def fitness(R, C, A):
    m, n = A.shape
    return sum([(R[i] & C[j] == C[j]) == A[i, j] for i in range(m) for j in range(n)])

def generate_solution(A, D):
    m, n = A.shape
    R = [random.choice(D) for _ in range(m)]
    C = [1 << random.randrange(8) for _ in range(n)]
    return R, C

def mutate_solution(R, C, D):
    m = len(R)
    n = len(C)
    i = random.randrange(m)
    R[i] = random.choice(D)
    j = random.randrange(n)
    C[j] = 1 << random.randrange(8)

def find_approximate_solution(A, D, population_size, num_iterations):
    m, n = A.shape
    population_R = []
    population_C = []
    for _ in range(population_size):
        R, C = generate_solution(A, D)
        population_R.append(R)
        population_C.append(C)
    fitness_values = [fitness(population_R[i], population_C[i], A) for i in range(population_size)]
    for iteration in range(num_iterations):
        for i in range(population_size):
            R_new = list(population_R[i])
            C_new = list(population_C[i])
            mutate_solution(R_new, C_new, D)
            fitness_new = fitness(R_new, C_new, A)
            if fitness_new > fitness_values[i]:
                population_R[i] = R_new
                population_C[i] = C_new
                fitness_values[i] = fitness_new
        best_index = max(range(population_size), key=lambda i: fitness_values[i])
        for i in range(population_size):
            if i != best_index:
                R_new = list(population_R[best_index])
                C_new = list(population_C[best_index])
                mutate_solution(R_new, C_new, D)
                fitness_new = fitness(R_new, C_new, A)
                if fitness_new > fitness_values[i]:
                    population_R[i] = R_new
                    population_C[i] = C_new
                    fitness_values[i] = fitness_new
    best_index = max(range(population_size), key=lambda i: fitness_values[i])
    best_R = population_R[best_index]
    best_C = population_C[best_index]
    return best_R[:], best_C[:]

A = ...
D = ...
x_vals = []
y_vals = []
z_vals = []
for population_size in range(1, 51, 5):
    for num_iterations in range(1, 61, 5):
        R, C = find_approximate_solution(A, D, population_size, num_iterations)
        m, n = A.shape
        F = np.empty(shape=(m,n),dtype=bool)
        for i in range(m):
            for j in range(n):
                F[i,j] =(R[i] & C[j] == C[j]) == A[i,j]
        me=np.sum(F)/(n*m)
        x_vals.append(population_size)
        y_vals.append(num_iterations)
        z_vals.append(me)

fig=plt.figure()
ax=plt.axes(projection='3d')
ax.scatter3D(x_vals,y_vals,z_vals,c=z_vals,cmap='Reds')
ax.set_xlabel('population size')
ax.set_ylabel('number of iterations')
ax.set_zlabel('approximation measure')
plt.show()

########## plotting #############################################




