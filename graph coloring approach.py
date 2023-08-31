
#-----------------------------------------------------------------------------------------
#-- run in:  colab ----------------------------

import numpy as np


def transitive_reduction(A):
    m, n = A.shape
    M = np.zeros((n, n), dtype=int)
    for j in range(n):
        for j1 in range(n):
            if j != j1 and np.all(A[:, j] >= A[:, j1]):
                M[j, j1] = 1
    M_trr = M.copy()
    for i in range(n):
        for j in range(n):
            if M_trr[i, j] == 1:
                for k in range(n):
                    if M_trr[j, k] == 1:
                        M_trr[i, k] = 0
    M_trr_sym = np.maximum(M_trr, M_trr.T)
    M_trr_sym_3circl = M_trr_sym.copy()
    for i in range(n):
        for k in range(n):
            if M_trr[i, k] == 1:
                for k1 in range(n):
                    if k1 != k and M_trr[i, k1] == 1:
                        M_trr_sym_3circl[k, k1] = 1
                        M_trr_sym_3circl[k1, k] = 1
    return M_trr, M_trr_sym_3circl



def randadj_matrix(n):
    adj_matrix = np.random.randint(2, size=(n, n))
    np.fill_diagonal(adj_matrix, 0)
    adj_matrix = np.triu(adj_matrix) + np.triu(adj_matrix, 1).T
    return adj_matrix
    
def greedy_coloring(adj_matrix, colors):
    n = len(adj_matrix)
    for i in range(n):
        if colors[i] == -1:
            used_colors = set()
            for j in range(n):
                if adj_matrix[i][j] == 1 and colors[j] != -1:
                    used_colors.add(colors[j])
            for color in range(n):
                if color not in used_colors:
                    colors[i] = color
                    break
    return max(colors) + 1, colors

def branch_and_bound(adj_matrix):
    n = len(adj_matrix)
    best_coloring = greedy_coloring(adj_matrix, [-1] * n)
    best_num_colors = best_coloring[0]
    stack = [(0, [-1] * n)]
    while stack:
        node, colors = stack.pop()
        if node == n:
            num_colors = max(colors) + 1
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_coloring = (num_colors, colors)
        else:
            used_colors = set()
            for j in range(n):
                if adj_matrix[node][j] == 1 and colors[j] != -1:
                    used_colors.add(colors[j])
            for color in range(best_num_colors):
                if color not in used_colors:
                    new_colors = colors.copy()
                    new_colors[node] = color
                    upper_bound, _ = greedy_coloring(adj_matrix, new_colors)
                    if upper_bound < best_num_colors:
                        stack.append((node + 1, new_colors))
    return best_coloring


def generate_random_matrix(m, n):
    A = np.random.choice([0, 1], size=(m, n), p=[0.7, 0.3])
    while (np.all(A == 0, axis=1).any() or np.all(A == 0, axis=0).any() or
           np.unique(A, axis=0).shape[0] != m or np.unique(A, axis=1).shape[1] != n):
        A = np.random.choice([0, 1], size=(m, n), p=[0.7, 0.3])
    return A

def generate_random_matrix_dom(m, n):
    A = np.random.choice([0, 1], size=(m, n), p=[0.7, 0.3])
    A[:, 0] = 1
    p_dom = 0.4
    for j in range(n - 1):
        if np.random.random() < p_dom:
            A[:, j] = np.maximum(A[:, j], A[:, j + 1])
    while (np.all(A == 0, axis=1).any() or np.all(A == 0, axis=0).any() or
           np.unique(A, axis=0).shape[0] != m or np.unique(A, axis=1).shape[1] != n):
        A = np.random.choice([0, 1], size=(m, n), p=[0.7, 0.3])
        A[:, 0] = 1
        for j in range(n - 1):
            if np.random.random() < p_dom:
                A[:, j] = np.maximum(A[:, j], A[:, j + 1])
    return A



#---finish
from z3 import*

def permute_list(l):
    unique_l = list(set(l))
    permutes = []
    for p in permutations(unique_l):
        d = dict(zip(unique_l, p))
        permutes.append([d[i] for i in l])
    return permutes


def CSP(A, D, coloring, bit_length=17):  

    Permuted_coloring = permute_list(coloring[1])
    Permuted_1s = [[2 ** p - 1 for p in perm] for perm in Permuted_coloring]

    defa = 2 ** bit_length - 1
    m, n = A.shape
    s = Solver()
    s.set("model.completion", True)
    s.set("timeout", 8888)

    R = [BitVec(f"rs_{i + 1}", bit_length) for i in range(m)]
    C = [BitVec(f"cs_{j + 1}", bit_length) for j in range(n)]
    
    Constr_D = [Or([r == d for d in D]) for r in R]

    Constr_P = Or([And([C[j] & perm[j] == perm[j] for j in range(n)]) for perm in Permuted_1s])
    
    Constr_C = [C[j] == reduce(lambda x,y: x & y, [R[i] for i in range(m) if A[i, j] == 1], defa) for j in range(n)]

    Constr_npermit = [R[i] & C[j] != C[j] for i in range(m) for j in range(n) if A[i, j] == 0]    

    s.add(Constr_D + Constr_P + Constr_C + Constr_npermit)

    return s.check()


#make D---
D = [7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38,
              70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56,
              88, 152, 104, 168, 200, 112, 176, 208, 224]
def concat_two_codes(codes2,codes1,bit_len_2,bit_len_1):
    shifted_codes2 = [c2 << bit_len_1 for c2 in codes2]
    codes2_con_codes1 = [sh_c2 + c1 for sh_c2 in shifted_codes2 for c1 in codes1]
    return codes2_con_codes1
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
#--------

#---------


#execute complete -----
m, n = 22, 18 
A = generate_random_matrix_dom(m, n)
D = concat_two_codes(D,D,8,8)
H = get_H(2, 1)  
D = concat_two_codes(H,D,2,16)

coloring = branch_and_bound(branch_and_bound(M_trr_sym_3circl)[1])
CSP(A, D, coloring, bit_length=17)
#----------------------

    
"""    
# Example instances with execution
for m in range(7, 22, 3):
    n = m - 3
    A = generate_random_matrix_dom(m, n)

    M_trr, M_trr_sym_3circl = transitive_reduction(A)
    print(A)
    print("\n",M_trr_sym_3circl)
    coloring = branch_and_bound(M_trr_sym_3circl)

    print("Minimum number of colors:", coloring[0])
    print("Greedy  number of colors:", greedy_coloring(M_trr_sym_3circl, colors = [-1] * n))
    print("Color assignment:", coloring[1])

"""



#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
    
    
    
    
    
    
  
