
#-----------------------------------------------------------------------------------------
#-- run in:  https://www.onlinegdb.com/online_python_compiler ----------------------------

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
    p_dom = 0.4
    for j in range(n - 1):
        if np.random.random() < p_dom:
            A[:, j] = np.maximum(A[:, j], A[:, j + 1])
    while (np.all(A == 0, axis=1).any() or np.all(A == 0, axis=0).any() or
           np.unique(A, axis=0).shape[0] != m or np.unique(A, axis=1).shape[1] != n):
        A = np.random.choice([0, 1], size=(m, n), p=[0.7, 0.3])
        for j in range(n - 1):
            if np.random.random() < p_dom:
                A[:, j] = np.maximum(A[:, j], A[:, j + 1])
    return A

    
    
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

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
    
    
    
    
    
    
  
